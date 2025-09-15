"""
app.py – FastAPI application for the Property Prediction API

Production-ready features:
- Structured JSON logging
- Config via environment variables (pydantic-settings)
- Lifespan-managed model & data loading (no heavy work at import time)
- Dependency injection for models, preprocessors, and static data
- Centralized exception handling with JSON errors
- Health, live, and ready probes
- CORS, GZip, Trusted Hosts, HTTPS redirect (optional)
- Request ID & timing middleware with templated-path logging
- Request body size limiting
- Simple thread-safe in-memory TTL cache for (subset of) training data
- Async-friendly endpoints (run heavy work in a threadpool)
- Optional Prometheus metrics (commented section)

Notes:
- Models and reports dirs are configurable via environment variables.
- For production, run with gunicorn + uvicorn workers:
  gunicorn -k uvicorn.workers.UvicornWorker rest_api.app:app --bind 0.0.0.0:8000 --workers 4
"""

from __future__ import annotations

import os
import json
import logging
import time
import uuid
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable, Dict, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from starlette.concurrency import run_in_threadpool

# === Domain imports ===
from rest_api.schemas.output_data_schemas import (
    EnrichedResponse,
    SimilarPropertiesResponse,
    PredictResponse,
    ExplainResponse,
)
from rest_api.schemas.request_data_schemas import AddressInput, PredictRequest, KnnRequest
from rest_api.services.amenities import enrich_property_data, load_static_datasets
from rest_api.services.model_loader import get_preprocessor
from rest_api.utils.s3_utils import load_latest_models
from rest_api.services.prediction import preprocess_input, run_prediction
from rest_api.services.knn import find_similar_properties
from rest_api.db.accessors.training_data_accessors import get_training_data
from rest_api.services.explain import get_report_data
from rest_api.utils.s3_utils import write_report

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# =============================
# Config (env-driven, typed)
# =============================
class Settings(BaseSettings):
    app_name: str = "Property Prediction API"
    env: str = "production"
    model_dir: str = os.getenv("MODEL_DIR", "../models")
    reports_dir: str = os.getenv("REPORTS_DIR", "../reports")
    reports_bucket: str = os.getenv("REPORTS_BUCKET", 'propmind-reports')
    aws_region: str = os.getenv('AWS_REGION', 'eu-west-2')
    allowed_hosts: str = "*"
    cors_origins: str = "*"
    enable_https_redirect: bool = False
    request_body_limit_mb: int = 10
    gzip_min_size: int = 500
    log_level: str = "INFO"
    training_cache_ttl_sec: int = 600
    ready_flag_file: str = ""
    enable_metrics: bool = False

    class Config:
        env_file = "../.env"
        extra = "ignore"


settings = Settings()


# =============================
# Structured logging
# =============================
class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": int(time.time() * 1000),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            payload["request_id"] = getattr(record, "request_id")
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = settings.log_level) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root.addHandler(handler)


configure_logging()
logger = logging.getLogger(__name__)


# =============================
# Thread-safe in-memory TTL cache
# =============================
@dataclass
class TTLCacheEntry:
    value: Any
    expires_at: float


class TTLCache:
    def __init__(self):
        self._store: Dict[str, TTLCacheEntry] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if not entry or entry.expires_at < time.time():
                self._store.pop(key, None)
                return None
            return entry.value

    def set(self, key: str, value: Any, ttl_sec: int) -> None:
        with self._lock:
            self._store[key] = TTLCacheEntry(value=value, expires_at=time.time() + ttl_sec)


training_cache = TTLCache()


# =============================
# App lifespan: load models once
# =============================
@dataclass
class AppResources:
    clf: Any
    regressors: Any
    preprocessor: Any
    static_data: Any


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up, loading models & datasets...")
    try:
        clf, regressors = load_latest_models()
        preprocessor = get_preprocessor()
        static_data = load_static_datasets()
        app.state.resources = AppResources(
            clf=clf, regressors=regressors, preprocessor=preprocessor, static_data=static_data
        )
        if settings.ready_flag_file:
            try:
                with open(settings.ready_flag_file, "w") as f:
                    f.write("ready")
            except Exception:
                logger.warning("Failed to write READY_FLAG_FILE", exc_info=True)
        logger.info("Models & datasets loaded.")
    except Exception:
        logger.exception("Fatal error loading models during startup.")
        raise

    yield

    logger.info("Shutting down application...")


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=JSONResponse,
    debug=True
)


# =============================
# Middleware
# =============================
@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next: Callable) -> Response:
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        logger.exception("Unhandled error", extra={"request_id": request_id})
        raise e
    duration_ms = int((time.time() - start) * 1000)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-ms"] = str(duration_ms)
    route = getattr(request.scope.get("route"), "path", request.url.path)
    logger.info(
        f"{request.method} {route} -> {response.status_code} in {duration_ms}ms",
        extra={"request_id": request_id},
    )
    return response


@app.middleware("http")
async def limit_body_size(request: Request, call_next: Callable) -> Response:
    max_bytes = settings.request_body_limit_mb * 1024 * 1024
    try:
        cl = int(request.headers.get("content-length", "0") or 0)
    except ValueError:
        cl = 0
    if cl > max_bytes:
        return JSONResponse(status_code=413, content={"detail": "Request body too large"})
    return await call_next(request)


origins = ["*"] if settings.cors_origins == "*" else [o.strip() for o in settings.cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=settings.gzip_min_size)

hosts = ["*"] if settings.allowed_hosts == "*" else [h.strip() for h in settings.allowed_hosts.split(",")]
app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)

if settings.enable_https_redirect:
    from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
    app.add_middleware(HTTPSRedirectMiddleware)


# =============================
# Error handlers
# =============================
class ErrorResponse(BaseModel):
    detail: str
    request_id: Optional[str] = None


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    rid = request.headers.get("X-Request-ID")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(detail=str(exc.detail), request_id=rid).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    rid = request.headers.get("X-Request-ID")
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "request_id": rid})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    rid = request.headers.get("X-Request-ID")
    logger.exception("Unhandled server error", extra={"request_id": rid})
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(detail="Internal server error", request_id=rid).model_dump(),
    )


# =============================
# Dependencies
# =============================
@dataclass
class ModelDeps:
    clf: Any
    regressors: Any
    preprocessor: Any
    static_data: Any


def get_deps(request: Request) -> ModelDeps:
    res: AppResources = request.app.state.resources
    return ModelDeps(
        clf=res.clf,
        regressors=res.regressors,
        preprocessor=res.preprocessor,
        static_data=res.static_data,
    )


# =============================
# Health endpoints
# =============================
@app.get("/ping", tags=["health"])
async def ping() -> PlainTextResponse:
    return PlainTextResponse(content="pong", status_code=200)


@app.get("/live", tags=["health"])
async def live() -> PlainTextResponse:
    return PlainTextResponse(content="live", status_code=200)


@app.get("/ready", tags=["health"])
async def ready(request: Request) -> PlainTextResponse:
    ok = hasattr(request.app.state, "resources") and request.app.state.resources is not None
    try:
        res: AppResources = request.app.state.resources
        ok = ok and res.clf is not None and res.preprocessor is not None
    except Exception:
        ok = False
    return PlainTextResponse(content="ready" if ok else "not-ready", status_code=200 if ok else 503)


# =============================
# Cached training data
# =============================
def cached_training_data(min_date: Optional[date] = None) -> pd.DataFrame:
    key = f"training:{min_date.isoformat() if min_date else 'all'}:cols=v2"
    cached = training_cache.get(key)
    if cached is not None:
        return cached

    df = get_training_data(min_date=min_date) if min_date else get_training_data()
    for c in ("lat", "lon"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    training_cache.set(key, df, ttl_sec=settings.training_cache_ttl_sec)
    return df


# =============================
# API endpoints
# =============================
@app.post("/enrich", response_model=EnrichedResponse, tags=["enrichment"])
async def enrich_from_address(
    request_body: AddressInput,
    deps: ModelDeps = Depends(get_deps),
) -> EnrichedResponse:
    try:
        row: Dict[str, Any] = enrich_property_data(request_body.address, deps.static_data)
        if len(row.keys()) < 8:
            return EnrichedResponse(
                matched=False,
                address=row.get("full_address"),
                postcode=row.get("postcode"),
                dist_to_park=row.get("dist_to_park"),
                dist_to_tube=row.get("dist_to_tube"),
                dist_to_school=row.get("dist_to_school"),
            )
        return EnrichedResponse(
            matched=True,
            address=row.get("full_address"),
            postcode=row.get("postcode"),
            built_date=row.get("built_date"),
            energy_eff=row.get("energy_eff"),
            tenure=row.get("tenure"),
            property_type=row.get("property_type"),
            built_form=row.get("built_form"),
            floor_area=row.get("floor_area"),
            floor_level=row.get("floor_level"),
            num_rooms=row.get("num_rooms"),
            dist_to_park=row.get("dist_to_park"),
            dist_to_tube=row.get("dist_to_tube"),
            dist_to_school=row.get("dist_to_school"),
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        logger.exception("Error in /enrich endpoint")
        raise HTTPException(status_code=500, detail="Failed to enrich property data")


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
async def predict(
    request_body: PredictRequest,
    deps: ModelDeps = Depends(get_deps),
) -> PredictResponse:
    def _run() -> PredictResponse:
        df = pd.DataFrame([request_body.model_dump()])
        feature_names = getattr(deps.clf, "feature_names_in_", None)
        X = preprocess_input(df, feature_names)
        pred = run_prediction(X, deps.clf, deps.regressors)
        return PredictResponse(
            pred_price=pred["pred_price"],
            pred_ppm2=pred["pred_ppm2"],
            price_low=pred["price_low"],
            price_high=pred["price_high"],
        )

    try:
        return await run_in_threadpool(_run)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        logger.exception("Unexpected server error during prediction")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/similar", response_model=SimilarPropertiesResponse, tags=["similar"])
async def similar_properties(
    request_body: KnnRequest,
    deps: ModelDeps = Depends(get_deps),
) -> SimilarPropertiesResponse:
    def _run() -> SimilarPropertiesResponse:
        training_df = cached_training_data(min_date=date.today() - timedelta(days=365))
        similar = find_similar_properties(
            request_body.model_dump(),
            training_df=training_df,
            preprocessor=deps.preprocessor,
        )
        return SimilarPropertiesResponse(
            estimated_price=similar["estimated_price"],
            estimated_price_per_m2=similar["estimated_price_per_m2"],
            comps_quality=similar["quality"],
            comps=similar["comps"],
            display_comps=similar["display_comps"],
        )

    try:
        return await run_in_threadpool(_run)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        logger.exception("Error in /similar endpoint")
        raise HTTPException(status_code=500, detail="Failed to compute similar properties")


@app.post("/explain", response_model=ExplainResponse, tags=["explain"])
async def explain(
    request_body: PredictRequest,
    deps: ModelDeps = Depends(get_deps),
) -> ExplainResponse:
    def _run() -> ExplainResponse:
        inputs = request_body.model_dump()
        training_df = cached_training_data()
        pred = get_report_data(inputs, training_df, deps)

        # Create filename with postcode + date
        postcode = inputs.get("postcode", "unknown").replace(" ", "")
        date_str = date.today().strftime("%Y%m%d")
        report_filename = f"report_{postcode}_{date_str}.html"
        s3_url = f"https://{settings.reports_bucket}.s3.{settings.aws_region}.amazonaws.com/{report_filename}"

        # Build full path
        report_path = os.path.join(settings.reports_dir, report_filename)
        write_report(pred, report_path)
        return ExplainResponse(
            report_url=s3_url,
            pred_price=pred["pred_price"],
            pred_ppm2=pred["pred_ppm2"],
            price_low=pred["price_low"],
            price_high=pred["price_high"],
            base_value=pred["base_value"],
            shap_values=pred["shap_values"],
            residuals=pred["residual_value"],
            display_comps=pred["display_comps"],
            comps_confidence=pred["comps_confidence"],
            prefix_trend=pred["prefix_trend"],
            city_trend=pred["city_trend"],
            nlp_analysis=pred["nlp_analysis"],
            property_info=pred["property_info"],
        )

    try:
        return await run_in_threadpool(_run)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        logger.exception("Error in /explain endpoint")
        raise HTTPException(status_code=500, detail="Failed to compute price explanation")
