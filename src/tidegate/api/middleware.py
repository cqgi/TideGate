from __future__ import annotations

import hashlib
import hmac
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import ulid
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from tidegate.config.holder import ConfigHolder
from tidegate.config.models import TenantConfig
from tidegate.core.errors import ErrorCategory, GatewayError
from tidegate.obs.logging import bind_request_id


@dataclass(frozen=True)
class _AuthEntry:
    tenant: TenantConfig
    expires_at: float


class _AuthCache:
    def __init__(self, capacity: int, ttl_s: float) -> None:
        self._capacity = capacity
        self._ttl_s = ttl_s
        self._entries: OrderedDict[str, _AuthEntry] = OrderedDict()

    def get(self, key_hash: str) -> TenantConfig | None:
        now = time.monotonic()
        entry = self._entries.get(key_hash)
        if entry is None:
            return None
        if entry.expires_at <= now:
            self._entries.pop(key_hash, None)
            return None
        self._entries.move_to_end(key_hash)
        return entry.tenant

    def put(self, key_hash: str, tenant: TenantConfig) -> None:
        self._entries[key_hash] = _AuthEntry(
            tenant=tenant, expires_at=time.monotonic() + self._ttl_s
        )
        self._entries.move_to_end(key_hash)
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_id = str(ulid.new())
        state = _scope_state(scope)
        state["request_id"] = request_id
        bind_request_id(request_id)

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Request-Id"] = request_id
                if "X-TideGate-Cache" not in headers:
                    headers["X-TideGate-Cache"] = "bypass"
                if "X-TideGate-Route" not in headers:
                    headers["X-TideGate-Route"] = "none"
            await send(message)

        await self._app(scope, receive, send_with_headers)


class AuthMiddleware:
    def __init__(self, app: ASGIApp, config: ConfigHolder) -> None:
        self._app = app
        self._config = config
        config_auth = config.current.server
        self._cache = _AuthCache(
            config_auth.auth_cache_size,
            config_auth.auth_cache_ttl_s,
        )
        self._cache_version = config.version

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        data_paths = {"/v1/chat/completions", "/v1/models"}
        if scope["type"] != "http" or scope["path"] not in data_paths:
            await self._app(scope, receive, send)
            return
        self._reset_cache_if_config_changed()

        headers = Headers(scope=scope)
        authorization = headers.get("Authorization", "")
        prefix = "Bearer "
        if not authorization.startswith(prefix):
            # SPEC-M0-3: data-plane auth failures use OpenAI-compatible 401 JSON.
            await self._send_error(
                scope,
                receive,
                send,
                GatewayError("invalid api key", ErrorCategory.CLIENT_ERROR, http_status=401),
            )
            return
        raw_key = authorization.removeprefix(prefix)
        key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        tenant = self._cache.get(key_hash) or self._find_tenant(key_hash)
        if tenant is None:
            # SPEC-M0-3: compare sha256(api_key) against tenant config.
            await self._send_error(
                scope,
                receive,
                send,
                GatewayError("invalid api key", ErrorCategory.CLIENT_ERROR, http_status=401),
            )
            return
        self._cache.put(key_hash, tenant)
        _scope_state(scope)["tenant"] = tenant
        await self._app(scope, receive, send)

    def _reset_cache_if_config_changed(self) -> None:
        if self._cache_version == self._config.version:
            return
        current = self._config.current.server
        # SPEC-M1-4: tenant reloads invalidate cached auth decisions immediately.
        self._cache = _AuthCache(current.auth_cache_size, current.auth_cache_ttl_s)
        self._cache_version = self._config.version

    def _find_tenant(self, key_hash: str) -> TenantConfig | None:
        for tenant in self._config.current.tenants:
            if hmac.compare_digest(key_hash, tenant.api_key_sha256):
                return tenant
        return None

    async def _send_error(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        exc: GatewayError,
    ) -> None:
        request_id = _scope_state(scope).get("request_id")
        headers = {
            "X-TideGate-Cache": "bypass",
            "X-TideGate-Route": "none",
        }
        if isinstance(request_id, str):
            headers["X-Request-Id"] = request_id
        response = JSONResponse(
            {
                "error": {
                    "message": exc.message,
                    "type": "authentication_error",
                    "code": exc.code,
                }
            },
            status_code=401,
            headers=headers,
        )
        await response(scope, receive, send)


def _scope_state(scope: Scope) -> dict[str, Any]:
    state = scope.setdefault("state", {})
    if not isinstance(state, dict):
        state = {}
        scope["state"] = state
    return state
