import torch
import cv2
import numpy as np

# ==========================================
# MODEL IMPORT
# ==========================================

from models.dehazeformer import dehazeformer_s

# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("[DEVICE]", device)

# ==========================================
# MODEL
# ==========================================

model = dehazeformer_s().to(device)

checkpoint = torch.load(
    "D:/OtonomProje/DehazeFormer/weights/dehazeformer-s.pth",
    map_location=device
)

# ==========================================
# CHECKPOINT FIX
# ==========================================

if "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

if "params" in checkpoint:
    checkpoint = checkpoint["params"]

new_checkpoint = {}

for k, v in checkpoint.items():

    k = k.replace("module.", "")

    new_checkpoint[k] = v

model.load_state_dict(
    new_checkpoint,
    strict=False
)

model.eval()

print("[MODEL LOADED]")

# ==========================================
# IMAGE LOAD
# ==========================================

img_bgr = cv2.imread(
    "D:/OtonomProje/DehazeFormer/test.png"
)

if img_bgr is None:

    print("[ERROR] IMAGE NOT FOUND")

    exit()

# ==========================================
# ORIGINAL SIZE
# ==========================================

orig_h, orig_w = img_bgr.shape[:2]

# ==========================================
# RGB
# ==========================================

img_rgb = cv2.cvtColor(
    img_bgr,
    cv2.COLOR_BGR2RGB
)

# ==========================================
# RESIZE
# ==========================================

img_rgb = cv2.resize(
    img_rgb,
    (512, 512)
)

# ==========================================
# NORMALIZE
# ==========================================

img_np = img_rgb.astype(np.float32) / 255.0

tensor = torch.from_numpy(
    img_np
).permute(2, 0, 1).unsqueeze(0).to(device)

# ==========================================
# INFERENCE
# ==========================================

with torch.no_grad():

    output = model(tensor)

# ==========================================
# DEBUG
# ==========================================

print("[OUTPUT MIN]", output.min().item())
print("[OUTPUT MAX]", output.max().item())

# ==========================================
# DYNAMIC RANGE FIX
# ==========================================

output = output.squeeze(0)

out_min = output.min()

out_max = output.max()

output = (output - out_min) / (
    out_max - out_min + 1e-8
)

output = output.unsqueeze(0)

# ==========================================
# TO IMAGE
# ==========================================

out_np = output.squeeze(0).permute(
    1,
    2,
    0
).cpu().numpy()

out_np = np.clip(
    out_np * 255.0,
    0,
    255
).astype(np.uint8)

# ==========================================
# RGB -> BGR
# ==========================================

out_bgr = cv2.cvtColor(
    out_np,
    cv2.COLOR_RGB2BGR
)

# ==========================================
# ORIGINAL SIZE
# ==========================================

out_bgr = cv2.resize(
    out_bgr,
    (orig_w, orig_h)
)

# ==========================================
# GAMMA CORRECTION
# ==========================================

# ==========================================
# POST-PROCESSING (main.py dehaze_frame ile aynı)
# ==========================================

# --- GAMMA CORRECTION ---
gamma = 1.15
table = np.array([
    ((i / 255.0) ** (1.0 / gamma)) * 255
    for i in np.arange(0, 256)
]).astype("uint8")
out_bgr = cv2.LUT(out_bgr, table)

# --- CLAHE CONTRAST ---
lab = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
l = clahe.apply(l)
out_bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

# --- SHARPEN ---
blur = cv2.GaussianBlur(out_bgr, (0, 0), 2.5)
out_bgr = cv2.addWeighted(out_bgr, 1.25, blur, -0.25, 0)

# --- BLEND WITH ORIGINAL ---
out_bgr = cv2.addWeighted(img_bgr, 0.15, out_bgr, 0.85, 0)

# --- FINAL CONTRAST ---
out_bgr = cv2.convertScaleAbs(out_bgr, alpha=1.04, beta=2)

# ==========================================
# CLAHE CONTRAST
# ==========================================

lab = cv2.cvtColor(
    out_bgr,
    cv2.COLOR_BGR2LAB
)

l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

l = clahe.apply(l)

lab = cv2.merge((l, a, b))

out_bgr = cv2.cvtColor(
    lab,
    cv2.COLOR_LAB2BGR
)

# ==========================================
# COLOR RECOVERY
# ==========================================

out_bgr = cv2.detailEnhance(
    out_bgr,
    sigma_s=8,
    sigma_r=0.12
)

# ==========================================
# EDGE PRESERVING
# ==========================================

out_bgr = cv2.edgePreservingFilter(
    out_bgr,
    flags=1,
    sigma_s=50,
    sigma_r=0.18
)

# ==========================================
# NATURAL SHARPEN
# ==========================================

blur = cv2.GaussianBlur(
    out_bgr,
    (0, 0),
    2
)

out_bgr = cv2.addWeighted(
    out_bgr,
    1.12,
    blur,
    -0.12,
    0
)

# ==========================================
# SATURATION BOOST
# ==========================================

hsv = cv2.cvtColor(
    out_bgr,
    cv2.COLOR_BGR2HSV
)

h, s, v = cv2.split(hsv)

s = cv2.multiply(
    s,
    1.15
)

s = np.clip(
    s,
    0,
    255
).astype(np.uint8)

hsv = cv2.merge((h, s, v))

out_bgr = cv2.cvtColor(
    hsv,
    cv2.COLOR_HSV2BGR
)

# ==========================================
# ADVANCED DENOISE
# ==========================================

out_bgr = cv2.bilateralFilter(
    out_bgr,
    d=7,
    sigmaColor=45,
    sigmaSpace=45
)

# ==========================================
# ORIGINAL DETAIL RESTORE
# ==========================================

out_bgr = cv2.addWeighted(
    img_bgr,
    0.42,
    out_bgr,
    0.58,
    0
)

# ==========================================
# FILMIC TONE
# ==========================================

out_bgr = cv2.convertScaleAbs(
    out_bgr,
    alpha=1.08,
    beta=4
)

# ==========================================
# FINAL SOFTEN
# ==========================================

soft = cv2.GaussianBlur(
    out_bgr,
    (3, 3),
    0
)

out_bgr = cv2.addWeighted(
    out_bgr,
    0.92,
    soft,
    0.08,
    0
)

# ==========================================
# SAVE
# ==========================================

cv2.imwrite(
    "D:/OtonomProje/DehazeFormer/result.png",
    out_bgr
)

# ==========================================
# SHOW
# ==========================================

cv2.imshow(
    "ORIGINAL",
    img_bgr
)

cv2.imshow(
    "DEHAZED",
    out_bgr
)

cv2.waitKey(0)

cv2.destroyAllWindows()

print("[DONE] result.png saved")