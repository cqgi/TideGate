from __future__ import annotations

import os

import redis.asyncio as redis
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from tidegate.config.models import CacheToggleConfig, TenantConfig
from tidegate.config.reloader import apply_reload, merge_tenants, publish_reload
from tidegate.config.tenant_store import TenantStore

router = APIRouter()


class TenantConfigUpdateIn(BaseModel):
    api_key_sha256: str
    plan: str
    policy: str
    cache: CacheToggleConfig = Field(default_factory=CacheToggleConfig)


def _authorized(request: Request) -> bool:
    token = request.headers.get("X-Admin-Token", "")
    expected = os.getenv("TIDEGATE_ADMIN_TOKEN", "")
    return bool(expected and token == expected)


def _auth_error() -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": "invalid admin token",
                "type": "authentication_error",
                "code": None,
            }
        },
        status_code=401,
    )


@router.post("/admin/config/reload")
async def reload_config(request: Request) -> JSONResponse:
    if not _authorized(request):
        return _auth_error()
    result = await apply_reload(request.app)
    if not result.ok:
        return JSONResponse(
            {"ok": False, "version": result.version, "error": result.error},
            status_code=422,
        )
    try:
        version = await publish_reload(request.app.state.redis)
    except redis.RedisError:
        # Local admin reload remains useful when the Redis broadcast bus is unavailable.
        version = request.app.state.config_holder.version + 1
    request.app.state.config_holder.replace(
        request.app.state.config_holder.current, version=version
    )
    return JSONResponse({"ok": True, "version": version})


@router.get("/admin/breakers")
async def breakers(request: Request) -> JSONResponse:
    if not _authorized(request):
        return _auth_error()
    return JSONResponse({"breakers": request.app.state.routing_state.snapshot()})


@router.get("/admin/tenants/{tenant_id}")
async def get_tenant(request: Request, tenant_id: str) -> JSONResponse:
    if not _authorized(request):
        return _auth_error()
    tenant = request.app.state.config_holder.current.tenant_by_id(tenant_id)
    if tenant is None:
        return JSONResponse(
            {
                "error": {
                    "message": "tenant not found",
                    "type": "invalid_request_error",
                    "code": None,
                }
            },
            status_code=404,
        )
    return JSONResponse({"tenant": tenant.model_dump()})


@router.put("/admin/tenants/{tenant_id}")
async def upsert_tenant(request: Request, tenant_id: str) -> JSONResponse:
    if not _authorized(request):
        return _auth_error()
    store: TenantStore = request.app.state.tenant_store
    if not store.enabled:
        return JSONResponse(
            {
                "error": {
                    "message": "tenant config database unavailable",
                    "type": "internal_error",
                    "code": None,
                }
            },
            status_code=503,
        )
    try:
        payload = TenantConfigUpdateIn.model_validate(await request.json())
        tenant = TenantConfig(id=tenant_id, **payload.model_dump())
    except (ValueError, ValidationError):
        return JSONResponse(
            {
                "error": {
                    "message": "request validation failed",
                    "type": "invalid_request_error",
                    "code": None,
                }
            },
            status_code=422,
        )
    error = _validate_tenant_references(request, tenant)
    if error is not None:
        return JSONResponse(
            {"error": {"message": error, "type": "invalid_request_error", "code": None}},
            status_code=422,
        )
    await store.upsert(tenant)
    db_tenants = await store.load_all()
    if db_tenants is None:
        return JSONResponse(
            {
                "error": {
                    "message": "tenant config database unavailable",
                    "type": "internal_error",
                    "code": None,
                }
            },
            status_code=503,
        )
    try:
        version = await publish_reload(request.app.state.redis)
    except redis.RedisError:
        version = request.app.state.config_holder.version + 1
    request.app.state.config_holder.replace(
        merge_tenants(request.app.state.config_holder.current, db_tenants),
        version=version,
    )
    return JSONResponse({"ok": True, "version": version})


def _validate_tenant_references(request: Request, tenant: TenantConfig) -> str | None:
    settings = request.app.state.config_holder.current
    if tenant.plan not in settings.quota_plans:
        return "unknown quota plan"
    if tenant.policy not in settings.policies:
        return "unknown policy"
    selected_point = tenant.cache.l2_operating_point
    if selected_point is None:
        return _validate_l2_runtime(request, tenant)
    point_names = {point.name for point in settings.cache.l2.operating_points}
    if selected_point not in point_names:
        return "unknown l2 operating point"
    return _validate_l2_runtime(request, tenant)


def _validate_l2_runtime(request: Request, tenant: TenantConfig) -> str | None:
    if tenant.cache.l2 and getattr(request.app.state, "embedding_service", None) is None:
        return "l2 cache runtime unavailable"
    return None
