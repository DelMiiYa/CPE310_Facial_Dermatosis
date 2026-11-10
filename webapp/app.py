"""DermaScan Face AI — FastAPI Application.

Connects the backend service layer (webapp.service), bilingual knowledge base
(webapp/knowledge_base.json), and static frontend (webapp/static/*) into a
production-ready web application conforming to the frozen contract.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from webapp.service import (
    ABSTENTION_THRESHOLD,
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    CALIBRATION_T_EFFNET,
    CALIBRATION_T_RESNET,
    CALIBRATION_W_EFFNET,
    CALIBRATION_W_RESNET,
    SYSTEM_NAME,
    WebAppInputError,
    analyze,
    get_service,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model singletons once at application startup into app.state.service."""
    svc = get_service()
    app.state.service = svc
    yield
    # Note: do NOT destroy the singleton _SERVICE_INSTANCE on lifespan exit
    # so that multiple TestClient(app) blocks in pytest can reuse the already-loaded
    # GPU models without reloading checkpoints repeatedly.


app = FastAPI(
    title=SYSTEM_NAME,
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(WebAppInputError)
async def webapp_input_error_handler(request: Request, exc: WebAppInputError) -> JSONResponse:
    """Handle custom application input errors with frozen error contract."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle FastAPI validation errors (e.g., missing form fields) with friendly bilingual contract."""
    return JSONResponse(
        status_code=400,
        content={
            "detail": "Missing or invalid upload file. Please select a valid image file (JPG, PNG, WEBP).",
            "detail_th": "ไม่พบไฟล์รูปภาพหรือข้อมูลที่ส่งมาไม่ถูกต้อง กรุณาเลือกไฟล์ภาพ (JPG, PNG, WEBP)",
            "error_code": "empty_file",
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fallback handler for unhandled internal exceptions."""
    logger.exception("Internal server error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error during image analysis.",
            "detail_th": "เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์ระหว่างการวิเคราะห์ภาพ",
            "error_code": "server_error",
        },
    )


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Return system status, calibration constants, device, and model readiness."""
    svc = get_service()
    models_loaded = svc.ensemble is not None
    kb_loaded = bool(svc.get_knowledge_base_entry("acne"))
    return {
        "status": "ok",
        "system_name": SYSTEM_NAME,
        "device": str(svc.device),
        "models_loaded": models_loaded,
        "knowledge_base_loaded": kb_loaded,
        "abstention_threshold": ABSTENTION_THRESHOLD,
        "calibration": {
            "t_resnet": CALIBRATION_T_RESNET,
            "t_effnet": CALIBRATION_T_EFFNET,
            "w_resnet": CALIBRATION_W_RESNET,
            "w_effnet": CALIBRATION_W_EFFNET,
        },
    }


@app.post("/api/analyze")
async def analyze_endpoint(
    file: Optional[UploadFile] = File(default=None),
    triage_mode: bool = Form(default=False),
    triage_mode_query: Optional[bool] = Query(default=None, alias="triage_mode"),
) -> JSONResponse:
    """Analyze an uploaded facial photo and return the frozen contract result.

    Accepts multipart/form-data with `file` and optional `triage_mode` (accepted
    either via Form or Query parameter).
    """
    effective_triage_mode: bool = bool(triage_mode or triage_mode_query)

    # 1. Check if file is provided
    if file is None:
        raise WebAppInputError(
            detail="No image file provided. Please upload a JPG, PNG, or WEBP photo.",
            detail_th="ไม่ได้เลือกไฟล์รูปภาพ กรุณาอัปโหลดไฟล์ภาพ JPG, PNG หรือ WEBP",
            error_code="empty_file",
            status_code=400,
        )

    # 2. Validate filename extension & MIME type
    filename = (file.filename or "").strip()
    ext = Path(filename).suffix.lower()
    content_type = (file.content_type or "").lower().strip()

    if ext and ext not in ALLOWED_EXTENSIONS:
        raise WebAppInputError(
            detail=f"Unsupported file extension '{ext}'. Allowed formats: JPG, PNG, WEBP.",
            detail_th=f"ไม่รองรับนามสกุลไฟล์ '{ext}' กรุณาใช้ไฟล์ JPG, PNG หรือ WEBP",
            error_code="invalid_file_type",
            status_code=415,
        )

    if content_type and content_type not in ALLOWED_MIME_TYPES and content_type != "application/octet-stream":
        raise WebAppInputError(
            detail=f"Unsupported content type '{content_type}'. Allowed formats: JPG, PNG, WEBP.",
            detail_th=f"ไม่รองรับประเภทไฟล์ '{content_type}' กรุณาใช้ไฟล์ JPG, PNG หรือ WEBP",
            error_code="invalid_file_type",
            status_code=415,
        )

    if not ext and content_type not in ALLOWED_MIME_TYPES:
        raise WebAppInputError(
            detail="Unsupported file type. Allowed formats: JPG, PNG, WEBP.",
            detail_th="รูปแบบไฟล์ไม่รองรับ กรุณาใช้ไฟล์ JPG, PNG หรือ WEBP",
            error_code="invalid_file_type",
            status_code=415,
        )

    # 3. Read raw image bytes
    image_bytes = await file.read()

    # 4. Run CPU/GPU pipeline in threadpool to prevent blocking the event loop
    try:
        result = await run_in_threadpool(analyze, image_bytes, effective_triage_mode)
    except WebAppInputError:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in analyze pipeline: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error during image analysis.",
                "detail_th": "เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์ระหว่างการวิเคราะห์ภาพ",
                "error_code": "server_error",
            },
        )

    # 5. Return frozen contract response
    return JSONResponse(status_code=200, content=result)


@app.get("/")
async def serve_index() -> FileResponse:
    """Serve the root single-page application frontend."""
    return FileResponse(STATIC_DIR / "index.html")


# Mount static assets directory
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount root static fallback for relative asset paths (style.css, logo.svg, app.js)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="root_static")
