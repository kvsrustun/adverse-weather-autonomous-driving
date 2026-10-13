"""
Bu betik:
1. Roboflow üzerinden veri setini indirir.
2. Sentetik sis (add_fog_natural) ve CLAHE ön işleme (dehaze_soft) uygular.
3. 7 sınıflı veri konfigürasyonunu (dataset.yaml) otomatik oluşturur.
4. YOLOv11 modelini 100 epoch boyunca sıfırdan eğitir ve en iyi ağırlıkları (best.pt) üretir.
"""

import os
import glob
import shutil
import cv2
import yaml
import numpy as np
from tqdm import tqdm
from roboflow import Roboflow
from ultralytics import YOLO

# ==============================================================================
# 1. ROBOFLOW VERİ İNDİRME AYARLARI
# ==============================================================================
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "DXBeLZ2wfjperfbOSiJh")
WORKSPACE_NAME   = "kevsers-workspace-d8njl"
PROJECT_NAME     = "otonom_zorlu_hava_surusu_v2"
VERSION          = 1

OUTPUT_DATASET_DIR = "./Soft_Mixed_Dataset"
TRAIN_IMG_DIR = os.path.join(OUTPUT_DATASET_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(OUTPUT_DATASET_DIR, "train", "labels")

# ==============================================================================
# 2. GÖRÜNTÜ ÖN İŞLEME VE VERİ ARTIRMA FONKSİYONLARI
# ==============================================================================
def dehaze_soft(image):
    """CLAHE kullanarak yerel kontrastı ve parlaklığı iyileştirir."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def add_fog_natural(image, intensity=0.6):
    """Doğal atmosferik sis saçılmasını simüle eden gradyan maske uygular."""
    row, col, ch = image.shape
    mask = np.zeros((row, col), dtype=np.float32)
    for i in range(row):
        mask[i, :] = (i / row) * intensity
    fog_color = (220, 220, 220)
    fog_img = image.copy()
    for c in range(3):
        fog_img[:, :, c] = image[:, :, c] * (1 - mask) + fog_color[c] * mask
    return fog_img.astype(np.uint8)

# ==============================================================================
# 3. VERİ HAZIRLIĞI PİPELİNE'I
# ==============================================================================
def prepare_dataset():
    print("📦 Roboflow üzerinden orijinal veriler indiriliyor...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
    dataset = project.version(VERSION).download("yolov11")

    # Klasörleri temizle ve yeniden oluştur
    shutil.rmtree(OUTPUT_DATASET_DIR, ignore_errors=True)
    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(TRAIN_LBL_DIR, exist_ok=True)

    raw_train_img = os.path.join(dataset.location, "train", "images")
    raw_train_lbl = os.path.join(dataset.location, "train", "labels")
    files = glob.glob(os.path.join(raw_train_img, "*.jpg"))

    print(f"🚀 {len(files)} adet görüntü işleniyor ve zenginleştiriliyor...")
    for f in tqdm(files):
        name = os.path.basename(f)
        lbl_f = os.path.join(raw_train_lbl, name.replace(".jpg", ".txt"))
        img = cv2.imread(f)
        if img is None:
            continue

        # 1. Ham Görüntü
        shutil.copy(f, os.path.join(TRAIN_IMG_DIR, "raw_" + name))
        if os.path.exists(lbl_f):
            shutil.copy(lbl_f, os.path.join(TRAIN_LBL_DIR, "raw_" + name.replace(".jpg", ".txt")))

        # 2. Ön İşlenmiş / Sis Eklenmiş Görüntü
        is_obstacle = False
        if os.path.exists(lbl_f):
            with open(lbl_f, 'r') as l:
                if "1 " in l.read():
                    is_obstacle = True

        proc_img = add_fog_natural(img) if is_obstacle else dehaze_soft(img)
        prefix = "fog_" if is_obstacle else "soft_"

        cv2.imwrite(os.path.join(TRAIN_IMG_DIR, prefix + name), proc_img)
        if os.path.exists(lbl_f):
            shutil.copy(lbl_f, os.path.join(TRAIN_LBL_DIR, prefix + name.replace(".jpg", ".txt")))

    print(f"✅ Eğitim verileri hazırlandı! Toplam Görüntü: {len(os.listdir(TRAIN_IMG_DIR))}")
    return dataset.location

# ==============================================================================
# 4. DATASET YAML OLUŞTURMA
# ==============================================================================
def create_yaml(dataset_location):
    classes = ['animal', 'obstacle', 'person', 'pothole', 'signal', 'speedbump', 'vehicle']
    
    yaml_path = "./data.yaml"
    data_config = {
        'train': os.path.abspath(TRAIN_IMG_DIR),
        'val': os.path.abspath(os.path.join(dataset_location, "valid", "images")),
        'nc': 7,
        'names': classes
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f)
    print(f"✅ YAML yapılandırma dosyası oluşturuldu: {yaml_path}")
    return yaml_path

# ==============================================================================
# 5. MODEL EĞİTİMİ (YOLOv11 SIFIRDAN EĞİTİM)
# ==============================================================================
def train_model(yaml_path):
    print("\n" + "=" * 60)
    print("🧠 YOLOv11 Modeli 100 Epoch Boyunca Eğitiliyor...")
    print("=" * 60)

    # Temel YOLOv11 mimarisiyle sıfırdan fine-tune başlatılır
    model = YOLO("yolo11n.pt") 

    # Eğitim hiperparametreleri (Tezde belirtilen 100 epoch ve 640x640 boyut)
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        name="Otonom_YOLOv11_Egitim",
        save=True
    )
    print("🎉 Eğitim tamamlandı! En iyi ağırlıklar runs/detect/Otonom_YOLOv11_Egitim/weights/best.pt dizinindedir.")

if __name__ == "__main__":
    dataset_loc = prepare_dataset()
    yaml_file   = create_yaml(dataset_loc)
    train_model(yaml_file)