"""HTTP API for connecting the QR plate workflow to an existing frontend."""

import base64
import binascii
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import BoundedSemaphore

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, RootModel
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from matcher import HybridImageMatcher

APP_ROOT = Path(__file__).resolve().parent
MODEL_PATHS = [
    ("YOLO26n", APP_ROOT / "models" / "yolo26n_cls_best.pt"),
    ("YOLO11n", APP_ROOT / "models" / "yolo11n_cls_best.pt"),
]
MAX_IMAGE_BYTES = 25 * 1024 * 1024
INFERENCE_WORKERS = max(
    1,
    int(os.getenv("QR_INFERENCE_WORKERS", str(min(4, os.cpu_count() or 1)))),
)
MAX_PENDING_PAIRS = max(
    INFERENCE_WORKERS,
    int(os.getenv("QR_MAX_PENDING_PAIRS", str(max(100, INFERENCE_WORKERS * 4)))),
)
INFERENCE_POOL = ThreadPoolExecutor(max_workers=INFERENCE_WORKERS, thread_name_prefix="qr-inference")
PENDING_SLOTS = BoundedSemaphore(MAX_PENDING_PAIRS)

CLASS_TO_REMARK = {
    "gap between plate and wall": "Gap between plate and wall; make sure the plate sits flush against the wall.",
    "nails not proper": "Only one nail is holding the QR code plate; secure it with at least two properly fixed nails.",
    "rescan scanify image": "Rescan the Scanify image clearly.",
    "stick plate properly": "Put the plate properly on the wall.",
    "write o for owner and r1, r2 for rent": "Mark owner plates as O and rental plates as R1 or R2, as applicable.",
}

allowed_origins = [
    origin.strip()
    for origin in os.getenv("QR_API_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]


class ImagePair(BaseModel):
    QRCodeImage: str
    HouseImage: str
    userId: int | str | None = None
    houseId: int | str | None = None
    QRStatus: int | None = None
    Remark: str | None = None


class BatchRequest(BaseModel):
    items: list[ImagePair] = Field(min_length=1, max_length=100)


class BatchRecordList(RootModel[list[ImagePair]]):
    root: list[ImagePair] = Field(min_length=1, max_length=100)


app = FastAPI(title="QR Plate Match and Inspect API", version="1.0.0")
WEB_ROOT = APP_ROOT / "web"
app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)
@lru_cache(maxsize=1)
def get_matcher():
    return HybridImageMatcher()


@lru_cache(maxsize=2)
def get_model(path: str):
    return YOLO(path)


def decode_image(value: str, field_name: str) -> Image.Image:
    """Decode raw base64 or a data:image/...;base64,... value."""
    encoded = "".join(value.strip().split())
    if encoded.lower().startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header.lower():
            raise HTTPException(400, f"{field_name} must be a base64 encoded image.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(400, f"{field_name} contains invalid base64 data.") from None
    if not raw:
        raise HTTPException(400, f"{field_name} is empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"{field_name} exceeds the 25 MB image limit.")
    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, f"{field_name} is not a supported image.") from None


def result(images: ImagePair, qr_status=None, remark=None, manual_review=False):
    """Return only the fields the existing frontend needs to update its record."""
    response = {
        "QRStatus": qr_status,
        "Remark": remark,
        "manual_review": manual_review,
    }
    if images.userId is not None:
        response["userId"] = images.userId
    if images.houseId is not None:
        response["houseId"] = images.houseId
    return response


def inspect_reference(reference: Image.Image, images: ImagePair):
    predictions = []
    for name, path in MODEL_PATHS:
        if not path.is_file():
            continue
        output = get_model(str(path)).predict(source=reference, device="cpu", verbose=False)[0]
        probabilities = output.probs.data.detach().cpu().numpy().astype(float)
        predictions.append(
            (name, {str(output.names[i]): float(probabilities[i]) for i in range(len(probabilities))})
        )
    if not predictions:
        raise HTTPException(503, "Inspector model weights are missing from the server.")

    classes = set().union(*(scores.keys() for _, scores in predictions))
    averages = {
        label: float(np.mean([scores.get(label, 0.0) for _, scores in predictions]))
        for label in classes
    }
    highest = max(averages, key=averages.get)
    winners = [max(scores, key=scores.get).strip().casefold() for _, scores in predictions]
    nail_manual = highest.strip().casefold() == "accept+" and any(w == "nails not proper" for w in winners)
    normalized = " ".join(highest.strip().split()).casefold()
    is_manual = (
        nail_manual
        or normalized == "stick plate properly"
        or any(word in normalized for word in ("shop", "store", "merchant", "metal"))
    )
    if is_manual:
        return result(images, qr_status=None, remark=images.Remark, manual_review=True)
    if normalized == "accept+":
        return result(images, qr_status=1, remark="")
    remark = CLASS_TO_REMARK.get(normalized, f"Inspector rejection: {highest.strip()}.")
    return result(images, qr_status=0, remark=remark)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def frontend():
    return FileResponse(WEB_ROOT / "index.html")


@app.post("/api/inspect")
def inspect_pair(images: ImagePair):
    future = submit_pair(images)
    if future is None:
        raise HTTPException(503, "Image processing is at capacity; retry this request shortly.")
    return future.result()


def submit_pair(images: ImagePair):
    """Submit one pair to the bounded shared inference pool, or decline if full."""
    if not PENDING_SLOTS.acquire(blocking=False):
        return None
    try:
        future = INFERENCE_POOL.submit(process_pair, images)
    except Exception:
        PENDING_SLOTS.release()
        raise
    future.add_done_callback(lambda _future: PENDING_SLOTS.release())
    return future


def process_pair(images: ImagePair):
    if not images.QRCodeImage.strip():
        raise HTTPException(422, "QRCodeImage must contain a base64 encoded image.")
    if not images.HouseImage.strip():
        raise HTTPException(422, "HouseImage must contain a base64 encoded image.")
    reference = decode_image(images.QRCodeImage, "QRCodeImage")
    property_image = decode_image(images.HouseImage, "HouseImage")
    try:
        match = get_matcher().verify_presence(reference, property_image)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Image matching failed: {exc}") from exc

    if match.get("needs_manual_review"):
        return result(images, qr_status=None, remark=images.Remark, manual_review=True)
    if not match.get("contains_reference"):
        return result(
            images,
            qr_status=0,
            remark="Reference plate or its surrounding background does not match the property image.",
        )
    # As in the UI workflow, inspect only the reference plate photo after a match.
    return inspect_reference(reference, images)


@app.post("/api/inspect/batch")
def inspect_batch(batch: BatchRequest | BatchRecordList):
    """Process up to 100 pairs concurrently within the shared inference limit."""
    items = batch.items if isinstance(batch, BatchRequest) else batch.root
    results = []
    submitted = []
    for index, images in enumerate(items):
        submitted.append((index, images, submit_pair(images)))

    for index, images, future in submitted:
        try:
            if future is None:
                raise HTTPException(503, "Image processing is at capacity; retry this item shortly.")
            item_result = future.result()
            item_result["success"] = True
        except HTTPException as exc:
            item_result = result(images, qr_status=None, remark=images.Remark)
            item_result.update({"success": False, "error": str(exc.detail)})
        except Exception as exc:
            item_result = result(images, qr_status=None, remark=images.Remark)
            item_result.update({"success": False, "error": f"Image analysis failed: {exc}"})
        item_result["index"] = index
        results.append(item_result)
    return {"results": results, "count": len(results)}


@app.get("/api/inspect/batch")
def batch_endpoint_help():
    """Explain the POST contract when this API URL is opened in a browser."""
    return {
        "message": "This endpoint processes images with POST. In Postman, choose Body > raw > JSON.",
        "method": "POST",
        "url": "/api/inspect/batch",
        "maximum_pairs_per_request": 100,
        "body_example": [
            {
                "houseId": 7044,
                "QRCodeImage": "<base64 reference plate image>",
                "HouseImage": "<base64 property image>",
            }
        ],
        "also_accepts_items_wrapper": True,
    }
