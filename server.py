import io
import base64
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torch
from PIL import Image
import time
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# -------------------------------
# Model load (global)
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Starting model on device:", device)

processor = TrOCRProcessor.from_pretrained("anuashok/ocr-captcha-v3")
model = VisionEncoderDecoderModel.from_pretrained("anuashok/ocr-captcha-v3").to(device)
model.eval()
torch.set_grad_enabled(False)
torch.set_num_threads(1)

# FP16 + Torch compile
if device == "cuda":
    model = model.half()
try:
    model = torch.compile(model, mode="reduce-overhead")
except Exception:
    pass

# Optional concurrency control
INFERENCE_CONCURRENCY = 8
inference_semaphore = asyncio.Semaphore(INFERENCE_CONCURRENCY)
EXECUTOR = ThreadPoolExecutor(max_workers=INFERENCE_CONCURRENCY)

# -------------------------------
# Helpers
# -------------------------------
BACKGROUND_CACHE = {}

def get_bg(size):
    if size not in BACKGROUND_CACHE:
        BACKGROUND_CACHE[size] = Image.new("RGBA", size, (255, 255, 255))
    return BACKGROUND_CACHE[size]

def _preprocess_image_bytes(contents: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(contents)).convert("RGBA")
    bg = get_bg(img.size)
    combined = Image.alpha_composite(bg, img).convert("RGB")
    return combined

def _predict_sync(contents: bytes, max_new_tokens: int = 16, num_beams: int = 1) -> str:
    """
    Optimized synchronous prediction with FP16, Torch compile, GPU stream
    """
    image = _preprocess_image_bytes(contents)
    pixel_values = processor(image, return_tensors="pt").pixel_values

    if device == "cuda":
        pixel_values = pixel_values.half()
    pixel_values = pixel_values.to(device)

    stream = torch.cuda.Stream() if device == "cuda" else None
    if stream:
        stream_context = torch.cuda.stream(stream)
    else:
        stream_context = torch.inference_mode()

    with stream_context, torch.inference_mode():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        if stream:
            stream.synchronize()

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return text

# -------------------------------
# Endpoint
# -------------------------------
@app.post("/solve")
async def solve(file: UploadFile = File(None), image_b64: str = Form(None)):
    """
    Accepts either a multipart/form file (field name "file") or a form field "image_b64" (base64 string).
    Returns JSON: {"text": "decoded_text", "ms": elapsed_milliseconds}.
    """
    if file is None and not image_b64:
        raise HTTPException(status_code=400, detail="Provide either 'file' (multipart) or 'image_b64' (form field).")

    if file is not None:
        contents = await file.read()
    else:
        try:
            contents = base64.b64decode(image_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="image_b64 not valid base64")

    async with inference_semaphore:
        loop = asyncio.get_running_loop()
        t0 = time.time()
        # Run the optimized synchronous prediction in a thread pool
        text = await loop.run_in_executor(EXECUTOR, _predict_sync, contents)
        elapsed = (time.time() - t0) * 1000

    return JSONResponse({"text": text, "ms": int(elapsed)})


# # parallelize this code
# import io
# import base64
# import asyncio
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.responses import JSONResponse
# from transformers import VisionEncoderDecoderModel, TrOCRProcessor
# import torch
# from PIL import Image
# import time

# app = FastAPI()

# # ---------- Model load (global; happens once on import) ----------
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Starting model on device:", device)

# processor = TrOCRProcessor.from_pretrained("anuashok/ocr-captcha-v3")
# model = VisionEncoderDecoderModel.from_pretrained("anuashok/ocr-captcha-v3").to(device)
# model.eval()
# torch.set_grad_enabled(False)
# torch.set_num_threads(1)

# # optional: speed tweaks for CUDA
# if device == "cuda":
#     torch.backends.cudnn.benchmark = True
#     try:
#         model.half()  # fp16 can reduce latency and memory on GPU (test for correctness)
#     except Exception:
#         pass

# # Optional concurrency control: tune to avoid OOM or contention
# INFERENCE_CONCURRENCY = 8
# inference_semaphore = asyncio.Semaphore(INFERENCE_CONCURRENCY)

# # ---------- helpers ----------
# def _preprocess_image_bytes(contents: bytes) -> Image.Image:
#     img = Image.open(io.BytesIO(contents)).convert("RGBA")
#     background = Image.new("RGBA", img.size, (255, 255, 255))
#     combined = Image.alpha_composite(background, img).convert("RGB")
#     return combined

# def _predict_sync(contents: bytes, max_new_tokens: int = 16, num_beams: int = 1) -> str:
#     """
#     Synchronous prediction function. It's CPU/GPU bound so we'll call it in an executor.
#     """
#     image = _preprocess_image_bytes(contents)
#     pixel_values = processor(image, return_tensors="pt").pixel_values
#     # if model was cast to half, cast inputs too (GPU only)
#     if device == "cuda" and next(model.parameters()).dtype == torch.float16:
#         pixel_values = pixel_values.half()
#     pixel_values = pixel_values.to(device)

#     with torch.inference_mode():
#         # generation params tuned for speed: small max_new_tokens, beams=1
#         generated_ids = model.generate(
#             pixel_values,
#             max_new_tokens=max_new_tokens,
#             num_beams=num_beams,
#         )
#     text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#     return text

# # ---------- endpoint ----------
# @app.post("/solve")
# async def solve(file: UploadFile = File(None), image_b64: str = Form(None)):
#     """
#     Accepts either a multipart/form file (field name "file") or a form field "image_b64" (base64 string).
#     Returns JSON: {"text": "decoded_text"}.
#     """
#     if file is None and not image_b64:
#         raise HTTPException(status_code=400, detail="Provide either 'file' (multipart) or 'image_b64' (form field).")

#     if file is not None:
#         contents = await file.read()
#     else:
#         try:
#             contents = base64.b64decode(image_b64)
#         except Exception:
#             raise HTTPException(status_code=400, detail="image_b64 not valid base64")

#     # Capacity control: prevent unlimited concurrent model calls (tune INFERENCE_CONCURRENCY)
    async with inference_semaphore:
        loop = asyncio.get_running_loop()
        t0 = time.time()
        # run the blocking work in a thread pool so we don't block the event loop
        text = await loop.run_in_executor(None, _predict_sync, contents)
        elapsed = (time.time() - t0) * 1000
        return JSONResponse({"text": text, "ms": int(elapsed)})
