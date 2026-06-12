from __future__ import annotations

import os

import redis.asyncio as redis
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from tidegate.config.reloader import apply_reload, publish_reload

router = APIRouter()


@router.post("/admin/config/reload")
async def reload_config(request: Request) -> JSONResponse:
    token = request.headers.get("X-Admin-Token", "")
    expected = os.getenv("TIDEGATE_ADMIN_TOKEN", "")
    if not expected or token != expected:
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
    result = await apply_reload(request.app)
    if not result.ok:
        return JSONResponse(
            {"ok": False, "version": result.version, "error": result.error},
            status_code=422,
        )
    try:
        version = await publish_reload(request.app.state.redis)
    except redis.RedisError:
        # DECISION: local admin reload remains useful when Redis broadcast bus is unavailable.
        version = request.app.state.config_holder.version + 1
    request.app.state.config_holder.replace(
        request.app.state.config_holder.current, version=version
    )
    return JSONResponse({"ok": True, "version": version})
