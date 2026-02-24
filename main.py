from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

app = FastAPI()

# ---------- Model load (once at startup) ----------
DEVICE = "cpu"  # Render free typically CPU
MODEL_NAME = "openai/clip-vit-base-patch32"

clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# ---------- Prompts ----------
ROAD_ACCIDENT = [
    "car crash on highway",
    "vehicle collision at intersection",
    "motorcycle accident on road",
    "truck crash accident",
    "bus accident on street",
    "car rollover accident",
    "damaged car after collision",
]

FIRE_ACCIDENT = [
    "building on fire with smoke",
    "car on fire accident",
    "house fire emergency",
    "kitchen fire accident",
    "warehouse fire with flames",
]

FALL_ACCIDENT = [
    "person falling down stairs",
    "elderly person slipped and fell",
    "person lying on ground injured",
    "human collapse emergency",
    "person slipped on wet floor",
]

ROAD_NORMAL = [
    "normal traffic on road",
    "cars driving safely",
    "parked vehicles roadside",
    "traffic jam without accident",
    "empty highway road",
]

FIRE_NORMAL = [
    "cooking on gas stove",
    "candle burning normally",
    "barbecue cooking flame",
    "steam from food cooking",
    "fireplace controlled fire",
    # normal home negatives
    "normal home interior living room no fire",
    "normal bedroom interior no smoke no fire",
    "normal kitchen interior no flames no fire",
    "warm indoor lighting in home no fire",
    "sunlight through window in home normal no fire",
]

FALL_NORMAL = [
    "person walking normally",
    "people standing and talking",
    "person sitting on chair",
    "child playing safely",
    "people exercising safely",
    # portrait/selfie negatives
    "portrait photo of a person normal",
    "selfie photo normal",
    "person posing for camera normal",
]

ALL_PROMPTS = ROAD_ACCIDENT + FIRE_ACCIDENT + FALL_ACCIDENT + ROAD_NORMAL + FIRE_NORMAL + FALL_NORMAL
ACC_LEN = len(ROAD_ACCIDENT) + len(FIRE_ACCIDENT) + len(FALL_ACCIDENT)

# Decision settings (যেটা তোমার আগের টেস্টে কাজ করেছে)
THRESHOLD = 0.30
MARGIN = 0.05

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}

def classify(img: Image.Image):
    inputs = clip_processor(text=ALL_PROMPTS, images=img, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = clip_model(**inputs)

    probs = torch.softmax(outputs.logits_per_image[0], dim=0).detach().cpu().numpy()

    accident_probs = probs[:ACC_LEN]
    normal_probs = probs[ACC_LEN:]

    best_acc_prob = float(np.max(accident_probs))
    best_norm_prob = float(np.max(normal_probs))

    # default: normal
    label = "normal"
    subtype = None
    score = best_norm_prob

    # accident only if strong enough
    if best_acc_prob >= THRESHOLD and (best_acc_prob - best_norm_prob) >= MARGIN:
        label = "accident"
        score = best_acc_prob
        idx = int(np.argmax(accident_probs))

        if idx < len(ROAD_ACCIDENT):
            subtype = "road"
        elif idx < len(ROAD_ACCIDENT) + len(FIRE_ACCIDENT):
            subtype = "fire"
        else:
            subtype = "fall"

    return label, subtype, score

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
    except Exception:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    label, subtype, score = classify(img)
    return {"label": label, "subtype": subtype, "score": float(score), "w": w, "h": h}
