from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io

from ultralytics import YOLO

app = FastAPI()

# --------- Model Load (CPU-friendly) ----------
# YOLOv8n is small and works on free tiers
yolo = YOLO("yolov8n.pt")

# --------- Heuristic labels (simple but practical) ----------
# We will detect common objects and decide:
# - road accident: vehicles + "crash cues" (many vehicles close) OR vehicle + person on road area
# - fall accident: person detected and image likely indoor + person low/lying (approx using bbox position)
# - fire accident: detect smoke/fire is not in coco labels; so we use color/smoke heuristic (bright orange/red + gray)
#
# This is not perfect, but it's stable + fast + deployable for beginner project.

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
PERSON_CLASS = "person"

def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def fire_heuristic(rgb: np.ndarray) -> float:
    """
    Returns a fire-likeness score (0..1) using simple color cues:
    - orange/red dominance for flames
    - gray-ish dominance for smoke
    """
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # flame-ish: high R, medium G, low B
    flame = (r > 0.6) & (g > 0.3) & (b < 0.35) & (r - b > 0.35)

    # smoke-ish: low saturation gray
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    sat = maxc - minc
    smoke = (maxc > 0.4) & (maxc < 0.85) & (sat < 0.10)

    flame_ratio = float(flame.mean())
    smoke_ratio = float(smoke.mean())

    # weighted score
    score = min(1.0, (flame_ratio * 8.0) + (smoke_ratio * 3.0))
    return score

def decide_label(objects, rgb: np.ndarray, w: int, h: int):
    """
    objects: list of dict {name, conf, x1,y1,x2,y2}
    """
    names = [o["name"] for o in objects]
    confs = [o["conf"] for o in objects]

    # --- Fire ---
    fire_score = fire_heuristic(rgb)
    if fire_score >= 0.55:
        return "accident", "fire", float(fire_score)

    # --- Road accident heuristic ---
    vehicle_objs = [o for o in objects if o["name"] in VEHICLE_CLASSES and o["conf"] >= 0.35]
    person_objs = [o for o in objects if o["name"] == PERSON_CLASS and o["conf"] >= 0.35]

    # many vehicles in one frame => possible crash / traffic incident
    if len(vehicle_objs) >= 3:
        score = 0.55 + min(0.35, 0.08 * (len(vehicle_objs) - 3))
        return "accident", "road", float(score)

    # vehicle + person very near bottom half => possible hit/fall on road
    if len(vehicle_objs) >= 1 and len(person_objs) >= 1:
        # check if any person bbox is low (near ground)
        for p in person_objs:
            y2 = p["y2"]
            if y2 > 0.85 * h:
                return "accident", "road", 0.58

    # --- Fall heuristic ---
    # Person bbox very wide + low => may indicate lying/collapsed
    if len(person_objs) >= 1:
        best = max(person_objs, key=lambda x: x["conf"])
        x1, y1, x2, y2 = best["x1"], best["y1"], best["x2"], best["y2"]
        bw, bh = (x2 - x1), (y2 - y1)

        # collapsed-like: bbox wide compared to height OR very low to ground
        if (bw > 1.2 * bh and y2 > 0.75 * h) or (y2 > 0.9 * h):
            return "accident", "fall", float(max(0.52, best["conf"]))

    # --- Normal ---
    # If vehicles exist but low count, assume normal traffic
    # If person exists but not collapsed-like, normal
    return "normal", None, 0.50

@app.get("/health")
def health():
    return {"status": "ok", "model": "yolov8n.pt", "mode": "lightweight"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        rgb = pil_to_np(img)
    except Exception:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # YOLO inference
    try:
        results = yolo.predict(rgb, imgsz=640, conf=0.25, verbose=False)
    except Exception as e:
        return JSONResponse({"error": f"Model inference failed: {str(e)}"}, status_code=500)

    r0 = results[0]
    objects = []
    if r0.boxes is not None and len(r0.boxes) > 0:
        for b in r0.boxes:
            cls_id = int(b.cls[0])
            name = r0.names.get(cls_id, str(cls_id))
            conf = float(b.conf[0])
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            objects.append({"name": name, "conf": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    label, subtype, score = decide_label(objects, rgb, w, h)

    return {
        "label": label,
        "subtype": subtype,
        "score": float(score),
        "w": w,
        "h": h,
        "detections": [{"name": o["name"], "conf": o["conf"]} for o in sorted(objects, key=lambda x: -x["conf"])[:10]]
    }
