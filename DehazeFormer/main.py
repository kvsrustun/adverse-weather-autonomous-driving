import carla
import cv2
import numpy as np
import torch
import collections
import random
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ultralytics import YOLO
from models.dehazeformer import dehazeformer_s

# =====================================================
# CONFIG
# =====================================================

CFG = {
    "SPEED_NORMAL":        30.0,
    "SPEED_CAUTION":       15.0,
    "SPEED_SLOW":          10.0,
    "SPEED_CREEP":          5.0,

    "PID_KP":               0.32,
    "PID_KI":               0.006,
    "PID_KD":               0.10,

    "STEER_KP":             0.80,
    "STEER_KI":             0.005,
    "STEER_KD":             0.15,

    "LIDAR_EMERGENCY":      5.0,
    "LIDAR_BRAKE_HARD":    12.0,
    "LIDAR_BRAKE_SOFT":    20.0,
    "LIDAR_CAUTION":       35.0,
    "LIDAR_EMA_ALPHA":      0.30,
    "LIDAR_LANE_HALF_W":    1.05,   
    "LIDAR_Z_MIN":         -0.3,
    "LIDAR_Z_MAX":          1.2,   
    "LIDAR_FRONT_MIN_X":    1.5,   

    "TTC_EMERGENCY":        1.0,
    "TTC_BRAKE_HARD":       2.5,
    "TTC_BRAKE_SOFT":       4.0,
    "TTC_CAUTION":          6.0,

    "PERSON_BOX_H_CLOSE":   28,
    "PERSON_BOX_H_MED":     18,

    "VEHICLE_BOX_H_CLOSE":  75,
    "VEHICLE_BOX_H_MED":    50,

    "CENTER_X_MIN":        185,
    "CENTER_X_MAX":        455,
    "RISK_IGNORE_OFFSIDE":  0.25,

    "STEER_SMOOTH":         0.60,
    "STEER_AHEAD_DIST":     3.5,
    "STEER_LOOKAHEAD_MIN":  5.0,
    "STEER_LOOKAHEAD_MAX":  8.0,

    "CTE_GAIN_LOW_SPD":     1.10,   
    "CTE_GAIN_HIGH_SPD":    0.80,   
    "CTE_MAX_CORRECTION":   0.55,   

    "THROTTLE_MAX":         0.50,
    "BRAKE_SMOOTH":         0.75,

    "NUM_VEHICLES":         30,
    "NUM_PEDESTRIANS":      60,
    "DELTA_SECONDS":        0.05,

    "W_LIDAR":              0.30,
    "W_TTC":                0.20,
    "W_YOLO_PERSON":        0.35,
    "W_YOLO_VEHICLE":       0.15,
    "W_TRAFFIC_LIGHT":      0.10,

    "YOLO_SKIP":             1,
    "DEHAZE_SKIP":           2,
    "LIDAR_MAX_PTS":      2500,
    "VALIDATION_SKIP":      10,
    "DISPLAY_W":           800,
    "DISPLAY_H":           450,
    "CLEANUP_INTERVAL":     30,
}

VALIDATION_CLASSES = {"PERSON", "VEHICLE", "SIGNAL"}

LABEL_MAP = {
    "animal":     "ANIMAL",
    "obstacle":   "OBSTACLE",
    "person":     "PERSON",
    "pothole":    "POTHOLE",
    "signal":     "SIGNAL",
    "speedbump":  "SPEEDBUMP",
    "vehicle":    "VEHICLE",
}

# =====================================================
# PID CONTROLLER
# =====================================================

class PIDController:

    def __init__(self, Kp=0.3, Ki=0.005, Kd=0.08):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0.0
        self.integral   = 0.0
        self._int_limit = 20.0

    def reset(self):
        self.prev_error = 0.0
        self.integral   = 0.0

    def step(self, error, dt):
        dt = max(dt, 0.001)
        self.integral += error * dt
        self.integral  = np.clip(self.integral, -self._int_limit, self._int_limit)
        derivative      = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

# =====================================================
# RISK SCORER
# =====================================================

class RiskScorer:

    def __init__(self):
        self.score    = 0.0
        self._history = collections.deque(maxlen=8)

    def _lidar_component(self, dist, ttc):
        d_score = 0.0
        if   dist < CFG["LIDAR_EMERGENCY"]:  d_score = 1.0
        elif dist < CFG["LIDAR_BRAKE_HARD"]: d_score = 0.85
        elif dist < CFG["LIDAR_BRAKE_SOFT"]: d_score = 0.60
        elif dist < CFG["LIDAR_CAUTION"]:    d_score = 0.30

        t_score = 0.0
        if   ttc < CFG["TTC_EMERGENCY"]:  t_score = 1.0
        elif ttc < CFG["TTC_BRAKE_HARD"]: t_score = 0.80
        elif ttc < CFG["TTC_BRAKE_SOFT"]: t_score = 0.55
        elif ttc < CFG["TTC_CAUTION"]:    t_score = 0.25

        return CFG["W_LIDAR"] * d_score + CFG["W_TTC"] * t_score

    def _yolo_component(self, box_details):
        score = 0.0
        for (lbl, cx, h) in box_details:
            in_center = CFG["CENTER_X_MIN"] < cx < CFG["CENTER_X_MAX"]

            if lbl == "PERSON":
                if h > CFG["PERSON_BOX_H_CLOSE"] and in_center:
                    score += CFG["W_YOLO_PERSON"]
                elif h > CFG["PERSON_BOX_H_MED"] and in_center:
                    score += CFG["W_YOLO_PERSON"] * 0.60
                elif h > CFG["PERSON_BOX_H_CLOSE"] and not in_center:
                    score += CFG["W_YOLO_PERSON"] * 0.15
                elif h > CFG["PERSON_BOX_H_MED"] and not in_center:
                    score += CFG["W_YOLO_PERSON"] * 0.05
                elif h > 12:
                    score += CFG["W_YOLO_PERSON"] * 0.02

            elif lbl == "VEHICLE":
                if h > CFG["VEHICLE_BOX_H_CLOSE"] and in_center:
                    score += CFG["W_YOLO_VEHICLE"]
                elif h > CFG["VEHICLE_BOX_H_MED"] and in_center:
                    score += CFG["W_YOLO_VEHICLE"] * 0.45
                elif h > CFG["VEHICLE_BOX_H_CLOSE"] and not in_center:
                    score += CFG["W_YOLO_VEHICLE"] * 0.02   
                elif h > CFG["VEHICLE_BOX_H_MED"] and not in_center:
                    score += 0.0                             

            elif lbl in ("OBSTACLE", "POTHOLE", "SPEEDBUMP"):
                score += 0.04
        return min(score, 1.0)

    def _tl_component(self, tl_state):
        if tl_state == "Red":    return CFG["W_TRAFFIC_LIGHT"]       
        if tl_state == "Yellow": return CFG["W_TRAFFIC_LIGHT"] * 0.3 
        return 0.0

    def compute(self, dist, ttc, box_details, detections, tl_state):
        raw = (
            self._lidar_component(dist, ttc) +
            self._yolo_component(box_details) +
            self._tl_component(tl_state)
        )
        raw = np.clip(raw, 0.0, 1.0)
        self._history.append(raw)
        self.score = float(np.mean(self._history))
        return self.score

# =====================================================
# DECISION ENGINE
# =====================================================

class DecisionEngine:

    _FOLLOW_ZONE_LO = CFG["LIDAR_BRAKE_HARD"]
    _FOLLOW_ZONE_HI = CFG["LIDAR_BRAKE_SOFT"]

    def __init__(self):
        self.last_risk      = 0.0
        self._tl_stop_lock  = False
        self._tl_lock_frame = 0   # kaç frame tl_stop'ta kaldık

    def _vehicle_decision(self, dist, ttc, box_details):
        # Sadece kendi şeridindeki araçlar (CENTER bölgesi + yeterli yükseklik)
        center_vehicle = any(
            lbl == "VEHICLE" and h > CFG["VEHICLE_BOX_H_CLOSE"]
            and CFG["CENTER_X_MIN"] < cx < CFG["CENTER_X_MAX"]
            for lbl, cx, h in box_details
        )

        # YAN ŞERİT kontrolü: merkez dışındaki araçları tespit et
        offside_vehicle = any(
            lbl == "VEHICLE"
            and not (CFG["CENTER_X_MIN"] < cx < CFG["CENTER_X_MAX"])
            for lbl, cx, h in box_details
        )

        if dist < CFG["LIDAR_EMERGENCY"] or ttc < CFG["TTC_EMERGENCY"]:
            return "hard_brake", 0.0
        if dist < self._FOLLOW_ZONE_LO or ttc < CFG["TTC_BRAKE_HARD"]:
            return "hard_brake", 0.0
        if dist < self._FOLLOW_ZONE_HI or ttc < CFG["TTC_BRAKE_SOFT"]:
            return "soft_brake", CFG["SPEED_CREEP"]

        # Yan şeritteki araç LIDAR mesafesini tetiklese bile caution'a düşme
        if (dist < CFG["LIDAR_CAUTION"] and center_vehicle) or \
           (dist < CFG["LIDAR_CAUTION"] and not offside_vehicle):
            return "caution", CFG["SPEED_CAUTION"]

        # Kendi şeridinde araç görülüyorsa (LIDAR'dan bağımsız YOLO kararı)
        if center_vehicle:
            return "caution", CFG["SPEED_CAUTION"]

        return "go", CFG["SPEED_NORMAL"]

    def _pedestrian_decision(self, box_details):
        best_mode = "go"
        best_spd  = CFG["SPEED_NORMAL"]

        PRIORITY = {"emergency": 0, "hard_brake": 1, "soft_brake": 2, "caution": 3, "go": 4}

        for lbl, cx, h in box_details:
            if lbl != "PERSON":
                continue
            in_center = CFG["CENTER_X_MIN"] < cx < CFG["CENTER_X_MAX"]

            if in_center and h > CFG["PERSON_BOX_H_CLOSE"]:
                # Sadece merkezdeki çok yakın insan → hard_brake
                mode, spd = "hard_brake", 0.0
            elif in_center and h > CFG["PERSON_BOX_H_MED"]:
                # Merkezdeki orta mesafe insan → yavaşla, durma
                mode, spd = "soft_brake", CFG["SPEED_SLOW"]
            elif not in_center and h > CFG["PERSON_BOX_H_CLOSE"]:
                # Yanda yakın insan → dikkat
                mode, spd = "caution", CFG["SPEED_CAUTION"]
            elif not in_center and h > CFG["PERSON_BOX_H_MED"]:
                # Yanda uzak insan → normal devam
                mode, spd = "go", CFG["SPEED_NORMAL"]
            else:
                # Küçük/uzak insan → yoksay, akışı bozma
                continue

            if PRIORITY[mode] < PRIORITY[best_mode]:
                best_mode, best_spd = mode, spd

        return best_mode, best_spd

    def _traffic_light_decision(self, tl_state):
        # Sadece gerçek kırmızıda dur
        if tl_state == "Red":
            self._tl_stop_lock  = True
            self._tl_lock_frame = 0
            return "tl_stop", 0.0

        # Kırmızı sonrası kilidi serbest bırak (Green veya Yellow geldi)
        if self._tl_stop_lock:
            self._tl_lock_frame += 1
            if self._tl_lock_frame >= 6:   # ~0.3 sn bekle, sonra aç
                self._tl_stop_lock  = False
                self._tl_lock_frame = 0
            else:
                return "tl_stop", 0.0      # kırmızıdan çıkış geçişi

        # Sarı = yavaşla ama durma
        if tl_state == "Yellow":
            return "caution", CFG["SPEED_CAUTION"]

        # Yeşil = devam
        return "go", CFG["SPEED_NORMAL"]

    def _hazard_decision(self, box_details):
        for lbl, cx, h in box_details:
            if lbl == "SPEEDBUMP":
                return "slow", CFG["SPEED_SLOW"]
            if lbl in ("POTHOLE", "OBSTACLE") and h > 35:
                return "caution", CFG["SPEED_CAUTION"]
        return "go", CFG["SPEED_NORMAL"]

    def decide(self, risk_score, dist, ttc, speed, box_details, tl_state):
        v_mode,  v_spd  = self._vehicle_decision(dist, ttc, box_details)
        p_mode,  p_spd  = self._pedestrian_decision(box_details)
        tl_mode, tl_spd = self._traffic_light_decision(tl_state)
        h_mode,  h_spd  = self._hazard_decision(box_details)

        if   risk_score > 0.85: rs_spd = CFG["SPEED_SLOW"]   # emergency değil, yavaşla
        elif risk_score > 0.55: rs_spd = CFG["SPEED_SLOW"]
        elif risk_score > 0.25: rs_spd = CFG["SPEED_CAUTION"]
        else:                   rs_spd = CFG["SPEED_NORMAL"]

        final_spd = min(v_spd, p_spd, tl_spd, h_spd, rs_spd)

        PRIORITY   = {"emergency": 0, "tl_stop": 1, "hard_brake": 2,
                      "soft_brake": 3, "slow": 4, "caution": 5, "go": 6}
        modes      = [v_mode, p_mode, tl_mode, h_mode]
        final_mode = min(modes, key=lambda m: PRIORITY.get(m, 99))

        if final_spd == 0.0 and final_mode == "go":
            final_mode = "caution"
            final_spd  = CFG["SPEED_SLOW"]

        self.last_risk = risk_score
        return final_mode, final_spd

# =====================================================
# LIDAR PROCESSOR  ← DEĞİŞTİ
# =====================================================

class LidarProcessor:

    def __init__(self):
        self.ema         = 50.0
        self.alpha       = CFG["LIDAR_EMA_ALPHA"]
        self.prev_dist   = 50.0
        self.ttc         = 99.0
        self.raw_dist    = 50.0
        self.ema_closing = 0.0

    def update(self, lidar_points, dt, vehicle, carla_map):
        raw = 50.0
        if lidar_points is not None and len(lidar_points) > 0:
            pts = lidar_points
            if len(pts) > CFG["LIDAR_MAX_PTS"]:
                idx = np.random.choice(len(pts), CFG["LIDAR_MAX_PTS"], replace=False)
                pts = pts[idx]

            v_trans = vehicle.get_transform()
            v_loc   = v_trans.location
            v_yaw   = math.radians(v_trans.rotation.yaw)

            # Şerit merkezi offset hesapla
            lane_offset_y = 0.0
            try:
                waypoint = carla_map.get_waypoint(
                    v_loc, project_to_road=True, lane_type=carla.LaneType.Driving
                )
                if waypoint is not None:
                    wp_loc = waypoint.transform.location
                    dx = wp_loc.x - v_loc.x
                    dy = wp_loc.y - v_loc.y
                    lane_offset_y = -dx * math.sin(v_yaw) + dy * math.cos(v_yaw)
            except Exception:
                pass

            half_w = CFG["LIDAR_LANE_HALF_W"]

            # ── TEMEL ŞERIT FİLTRESİ ──────────────────────────────────
            mask = (
                (pts[:, 0] > CFG["LIDAR_FRONT_MIN_X"]) &         
                (np.abs(pts[:, 1] - lane_offset_y) < half_w) &
                (pts[:, 2] > CFG["LIDAR_Z_MIN"]) &
                (pts[:, 2] < CFG["LIDAR_Z_MAX"])                  
            )
            front = pts[mask]

            # ── BİNA / DUVAR CLUSTER FİLTRESİ ────────────────────────
            # Kalan noktalarda Y yayılımı geniş + X yayılımı dar ise bina duvarıdır
            if len(front) > 5:
                y_range = np.ptp(front[:, 1])   # Y (sağ-sol) aralığı
                x_range = np.ptp(front[:, 0])   # X (ileri-geri) aralığı

                # Bina tespiti: Y yayılımı > 2.5 m VE X yayılımı < 1.5 m
                # → araç önündeki gerçek obje değil, yan duvar/bina
                if y_range > 2.5 and x_range < 1.5:
                    # Sadece şerit merkezine yakın noktaları tut
                    close_mask = np.abs(front[:, 1] - lane_offset_y) < 0.7
                    front = front[close_mask]

                # Ek kontrol: çok geniş Y ama düşük nokta sayısı → gürültü
                if y_range > 3.5 and len(front) < 15:
                    front = np.empty((0, 4), dtype=np.float32)

            if len(front) > 3:
                dists = np.sqrt(front[:, 0] ** 2 + front[:, 1] ** 2)
                raw = float(np.clip(np.percentile(dists, 5), 5.0, 50.0))

        self.raw_dist  = raw
        self.ema       = self.alpha * raw + (1.0 - self.alpha) * self.ema
        dist           = self.ema

        instant_closing  = (self.prev_dist - dist) / max(dt, 0.001)
        self.ema_closing = 0.3 * instant_closing + 0.7 * self.ema_closing
        self.prev_dist   = dist
        closing          = self.ema_closing

        self.ttc = dist / closing if closing > 0.3 else 99.0
        return dist, self.ttc

# =====================================================
# VALIDATION MANAGER
# =====================================================

class ValidationManager:

    def __init__(self):
        self.stats          = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in VALIDATION_CLASSES}
        self.precision      = {c: 0.0 for c in VALIDATION_CLASSES}
        self.recall         = {c: 0.0 for c in VALIDATION_CLASSES}
        self.mean_precision = 0.0
        self.mean_recall    = 0.0
        self.mean_accuracy  = 0.0

    def update(self, detections, gt_counts):
        for cls in VALIDATION_CLASSES:
            det = detections.get(cls, 0)
            gt  = gt_counts.get(cls, 0)
            tp  = min(det, gt)
            self.stats[cls]["tp"] += tp
            self.stats[cls]["fp"] += max(0, det - gt)
            self.stats[cls]["fn"] += max(0, gt - det)

        ps, rs = [], []
        for cls in VALIDATION_CLASSES:
            tp = self.stats[cls]["tp"]
            fp = self.stats[cls]["fp"]
            fn = self.stats[cls]["fn"]
            p  = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            self.precision[cls] = p * 100
            self.recall[cls]    = r * 100
            ps.append(p)
            rs.append(r)

        self.mean_precision = np.mean(ps) * 100
        self.mean_recall    = np.mean(rs) * 100
        self.mean_accuracy  = (self.mean_precision + self.mean_recall) / 2

    def get_gt(self, vehicle, world):
        gt      = {c: 0 for c in VALIDATION_CLASSES}
        v_trans = vehicle.get_transform()
        v_loc   = v_trans.location
        v_fwd   = v_trans.get_forward_vector()

        for actor in world.get_actors():
            try:
                if actor.id == vehicle.id:
                    continue
                a_loc = actor.get_location()

                is_traffic_light = "traffic_light" in actor.type_id
                max_dist = 60 if is_traffic_light else 45
                if v_loc.distance(a_loc) > max_dist:
                    continue

                dx     = a_loc.x - v_loc.x
                dy     = a_loc.y - v_loc.y
                length = math.sqrt(dx * dx + dy * dy) + 1e-6
                dot    = (v_fwd.x * dx + v_fwd.y * dy) / length

                threshold = 0.60 if is_traffic_light else 0.75
                if dot < threshold:
                    continue

                if   "walker.pedestrian" in actor.type_id: gt["PERSON"]  += 1
                elif "vehicle"           in actor.type_id: gt["VEHICLE"] += 1
                elif "traffic_light"     in actor.type_id: gt["SIGNAL"]  += 1
            except:
                pass
        return gt

    def final_report(self):
        print("\n" + "=" * 72)
        print("  OTONOM SÜRÜŞ VALİDASYON RAPORU")
        print("=" * 72)
        for cls in VALIDATION_CLASSES:
            tp = self.stats[cls]["tp"]
            fp = self.stats[cls]["fp"]
            fn = self.stats[cls]["fn"]
            p  = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            r  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r)  if (p + r)  > 0 else 0
            print(
                f"  {cls:<10} | "
                f"Precision: %{p:6.2f} | "
                f"Recall: %{r:6.2f} | "
                f"F1: %{f1:6.2f} | "
                f"TP:{tp}  FP:{fp}  FN:{fn}"
            )
        print(
            f"\n  Ortalama   | "
            f"Precision: %{self.mean_precision:6.2f} | "
            f"Recall: %{self.mean_recall:6.2f} | "
            f"Accuracy: %{self.mean_accuracy:6.2f}"
        )
        print("=" * 72)

    def save_chart(self, output_path="validation_report.png"):
        classes    = sorted(VALIDATION_CLASSES)
        precisions, recalls, f1s = [], [], []

        for cls in classes:
            tp = self.stats[cls]["tp"]
            fp = self.stats[cls]["fp"]
            fn = self.stats[cls]["fn"]
            p  = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            r  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r)  if (p + r)  > 0 else 0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

        labels     = classes + ["ORTALAMA"]
        precisions += [self.mean_precision]
        recalls    += [self.mean_recall]
        f1s        += [(self.mean_precision + self.mean_recall) / 2]

        x, width = np.arange(len(labels)), 0.25

        BG, ACCENT1, ACCENT2, ACCENT3 = "#0d1117", "#00e5ff", "#69ff47", "#ff6b6b"
        GRID, TEXT = "#1f2937", "#e6edf3"

        fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
        ax.set_facecolor(BG)

        for bars, data, color, label in [
            (ax.bar(x - width, precisions, width, color=ACCENT1, alpha=0.88, zorder=3), precisions, ACCENT1, "Precision"),
            (ax.bar(x,          recalls,    width, color=ACCENT2, alpha=0.88, zorder=3), recalls,    ACCENT2, "Recall"),
            (ax.bar(x + width,  f1s,        width, color=ACCENT3, alpha=0.88, zorder=3), f1s,        ACCENT3, "F1 Score"),
        ]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2,
                        f"{h:.1f}%", ha="center", va="bottom",
                        fontsize=8.5, color=TEXT, fontweight="bold")

        ax.axvspan(len(labels) - 1 - 0.45, len(labels) - 1 + 0.45,
                   color="#ffffff", alpha=0.04, zorder=1)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11, color=TEXT, fontweight="bold")
        ax.set_ylim(0, 115)
        ax.set_yticks(range(0, 101, 20))
        ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)], color=TEXT, fontsize=9)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", length=0)
        ax.set_title("OTONOM SÜRÜŞ — VALİDASYON RAPORU",
                     color=TEXT, fontsize=14, fontweight="bold", pad=18)
        ax.legend(
            handles=[
                mpatches.Patch(color=ACCENT1, label="Precision"),
                mpatches.Patch(color=ACCENT2, label="Recall"),
                mpatches.Patch(color=ACCENT3, label="F1 Score"),
            ],
            facecolor="#161b22", edgecolor=GRID,
            labelcolor=TEXT, fontsize=9, loc="upper right"
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"[CHART] Grafik kaydedildi → {output_path}")

        chart_img = cv2.imread(output_path)
        if chart_img is not None:
            cv2.imshow("Validation Report", chart_img)
            print("[CHART] Kapatmak için herhangi bir tuşa basın...")
            cv2.waitKey(0)
            cv2.destroyWindow("Validation Report")

# =====================================================
# TELEMETRY LOGGER
# =====================================================

class Telemetry:

    def __init__(self):
        self.log = []

    def record(self, frame, speed, risk, ttc, dist,
               mode, tl_state, throttle, brake, steer):
        self.log.append({
            "frame":    frame,
            "speed":    round(speed, 1),
            "risk":     round(risk, 3),
            "ttc":      round(ttc, 2),
            "dist":     round(dist, 2),
            "mode":     mode,
            "tl":       tl_state,
            "throttle": round(throttle, 3),
            "brake":    round(brake, 3),
            "steer":    round(steer, 3),
        })

    def summary(self):
        if not self.log:
            return
        speeds = [r["speed"] for r in self.log]
        risks  = [r["risk"]  for r in self.log]
        emerg  = sum(1 for r in self.log if r["mode"] == "emergency")
        print("\n" + "=" * 72)
        print("  TELEMETRY ÖZETİ")
        print("=" * 72)
        print(f"  Toplam kare      : {len(self.log)}")
        print(f"  Ort. hız         : {np.mean(speeds):.1f} km/h")
        print(f"  Maks hız         : {max(speeds):.1f} km/h")
        print(f"  Ort. risk skoru  : {np.mean(risks):.3f}")
        print(f"  Acil fren sayısı : {emerg}")
        print("=" * 72)

# =====================================================
# MAIN AGENT
# =====================================================

class AutonomousAgent:

    def __init__(self, model_path):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(60.0)
        self.world  = self.client.load_world('Town10HD')
        self.map    = self.world.get_map()

        weather = carla.WeatherParameters(
            cloudiness=70.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wetness=5.0,
            fog_density=38.0,
            fog_distance=45.0,
            fog_falloff=0.2,
            sun_altitude_angle=65.0,
            scattering_intensity=0.5,
            mie_scattering_scale=0.03,
            rayleigh_scattering_scale=0.0331,
        )
        self.world.set_weather(weather)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = CFG["DELTA_SECONDS"]
        self.world.apply_settings(settings)

        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)
        self.tm.set_global_distance_to_leading_vehicle(3.5)

        self.blueprints = self.world.get_blueprint_library()
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[DEVICE]", self.device)

        self.model = YOLO(model_path)
        print("[YOLO LOADED] — sınıflar:", self.model.names)

        # ---- DehazeFormer ----
        self.dehaze_model = dehazeformer_s().to(self.device)
        checkpoint = torch.load(
            "D:/OtonomProje/DehazeFormer/weights/dehazeformer-s.pth",
            map_location=self.device,
        )
        if "state_dict" in checkpoint: checkpoint = checkpoint["state_dict"]
        if "params"     in checkpoint: checkpoint = checkpoint["params"]
        new_ckpt = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        self.dehaze_model.load_state_dict(new_ckpt, strict=False)
        self.dehaze_model.eval()
        print("[DEHAZEFORMER LOADED]")

        # ---- Modüller ----
        self.speed_pid  = PIDController(CFG["PID_KP"], CFG["PID_KI"], CFG["PID_KD"])
        self.steer_pid  = PIDController(CFG["STEER_KP"], CFG["STEER_KI"], CFG["STEER_KD"])
        self.lidar_proc = LidarProcessor()
        self.risk_scorer= RiskScorer()
        self.decision   = DecisionEngine()
        self.validation = ValidationManager()
        self.telemetry  = Telemetry()

        # ---- Durum ----
        self.frame_count   = 0
        self.cam_queue     = collections.deque(maxlen=1)
        self.lidar_points  = None
        self.actors        = []
        self.prev_time     = time.time()

        self.throttle      = 0.0
        self.brake         = 0.0
        self.steer         = 0.0
        self.target_speed  = 0.0
        self.current_mode  = "go"
        self.tl_state      = "Green"
        self.dist          = 50.0
        self.ttc           = 99.0
        self.risk_score    = 0.0
        self._steer_smooth = 0.0

        self._last_detections   = {c: 0 for c in VALIDATION_CLASSES}
        self._last_box_details  = []
        self._last_label_counts = {}
        self._last_results      = None
        self._last_dehazed      = None
        self._last_signal_boxes = []

    # =====================================================
    # SETUP SENSORS
    # =====================================================

    def setup_sensors(self):
        bp    = self.blueprints.filter('model3')[0]
        spawn = random.choice(self.map.get_spawn_points())
        self.vehicle = self.world.spawn_actor(bp, spawn)
        self.actors.append(self.vehicle)

        cam_bp = self.blueprints.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '640')
        cam_bp.set_attribute('image_size_y', '360')
        cam_bp.set_attribute('fov', '100')
        self.camera = self.world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            attach_to=self.vehicle,
        )
        self.camera.listen(self.cam_queue.append)
        self.actors.append(self.camera)

        lidar_bp = self.blueprints.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '56000')
        lidar_bp.set_attribute('rotation_frequency', '20')
        self.lidar = self.world.spawn_actor(
            lidar_bp,
            carla.Transform(carla.Location(x=1.6, z=2.0)),
            attach_to=self.vehicle,
        )
        self.lidar.listen(self._lidar_cb)
        self.actors.append(self.lidar)

    # =====================================================
    # SPAWN WORLD
    # =====================================================

    def spawn_world_objects(self):
        num_v     = CFG["NUM_VEHICLES"]
        num_p     = CFG["NUM_PEDESTRIANS"]
        spawn_pts = self.map.get_spawn_points()
        veh_bps   = self.blueprints.filter('vehicle.*')

        for _ in range(num_v):
            try:
                npc = self.world.try_spawn_actor(
                    random.choice(veh_bps), random.choice(spawn_pts)
                )
                if npc:
                    npc.set_autopilot(True, self.tm.get_port())
                    self.actors.append(npc)
            except:
                pass

        walker_bps = self.blueprints.filter('walker.pedestrian.*')

        spawned_walkers = []
        for _ in range(num_p):
            try:
                loc = self.world.get_random_location_from_navigation()
                if not loc:
                    continue
                walker = self.world.try_spawn_actor(
                    random.choice(walker_bps), carla.Transform(loc)
                )
                if walker:
                    spawned_walkers.append(walker)
                    self.actors.append(walker)
            except:
                pass

        self.world.tick()

        for walker in spawned_walkers:
            try:
                ctrl = self.world.spawn_actor(
                    self.blueprints.find('controller.ai.walker'),
                    carla.Transform(), attach_to=walker,
                )
                ctrl.start()
                ctrl.go_to_location(self.world.get_random_location_from_navigation())
                ctrl.set_max_speed(1.0 + random.random() * 2.2)
                self.actors.append(ctrl)
            except:
                pass

        print(f"[WORLD] {num_v} araç + {len(spawned_walkers)} yaya oluşturuldu.")

    # =====================================================
    # LIDAR CALLBACK
    # =====================================================

    def _lidar_cb(self, data):
        self.lidar_points = np.frombuffer(
            data.raw_data, dtype=np.float32
        ).reshape((-1, 4))

    # =====================================================
    # DEHAZEFORMER
    # =====================================================

    def dehaze_frame(self, frame):
        orig_h, orig_w = frame.shape[:2]
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (320, 320))
        img_np  = resized.astype(np.float32) / 255.0
        tensor  = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            output = self.dehaze_model(tensor)

        output  = output.squeeze(0)
        out_min = output.min()
        out_max = output.max()
        output  = (output - out_min) / (out_max - out_min + 1e-8)
        out_np  = output.permute(1, 2, 0).cpu().numpy()
        out_np  = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
        out_bgr = cv2.resize(out_bgr, (orig_w, orig_h))

        gamma = 1.15
        table = np.array([
            ((i / 255.0) ** (1.0 / gamma)) * 255
            for i in np.arange(0, 256)
        ]).astype("uint8")
        out_bgr = cv2.LUT(out_bgr, table)

        lab     = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l       = clahe.apply(l)
        out_bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        blur    = cv2.GaussianBlur(out_bgr, (0, 0), 2.5)
        out_bgr = cv2.addWeighted(out_bgr, 1.25, blur, -0.25, 0)

        out_bgr = cv2.addWeighted(frame, 0.15, out_bgr, 0.85, 0)
        out_bgr = cv2.convertScaleAbs(out_bgr, alpha=1.04, beta=2)
        return out_bgr

    # =====================================================
    # SIGNAL COLOR ANALYSIS (HSV)
    # =====================================================

    def _analyze_signal_color(self, frame, x1, y1, x2, y2):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return "Unknown"

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "Unknown"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask_r1  = cv2.inRange(hsv, np.array([0,   80, 80]), np.array([12,  255, 255]))
        mask_r2  = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
        red_px   = cv2.countNonZero(mask_r1) + cv2.countNonZero(mask_r2)

        mask_g   = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255]))
        green_px = cv2.countNonZero(mask_g)

        mask_y    = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))
        yellow_px = cv2.countNonZero(mask_y)

        total = red_px + green_px + yellow_px
        if total < 30:
            return "Unknown"

        if red_px    > green_px and red_px    > yellow_px: return "Red"
        if yellow_px > red_px   and yellow_px > green_px:  return "Yellow"
        if green_px  > red_px:                              return "Green"
        return "Unknown"

    # =====================================================
    # TRAFFIC LIGHT: CARLA API + YOLO FUSION
    # =====================================================

    def _check_traffic_light(self, frame=None, signal_boxes=None):
        # ── CARLA API ────────────────────────────────────────────────────────
        carla_state = "Green"
        if self.vehicle.is_at_traffic_light():
            try:
                tl = self.vehicle.get_traffic_light()
                if tl is not None:
                    tl_loc  = tl.get_location()
                    v_loc   = self.vehicle.get_location()
                    v_fwd   = self.vehicle.get_transform().get_forward_vector()
                    dx      = tl_loc.x - v_loc.x
                    dy      = tl_loc.y - v_loc.y
                    dist_tl = math.sqrt(dx * dx + dy * dy)
                    dot     = (v_fwd.x * dx + v_fwd.y * dy) / (dist_tl + 1e-6)

                    # SADECE gerçek Red state'de dur, Yellow → CARLA'ya bırak ama "Yellow" olarak işaretle
                    # Green / Unknown / Off → Green say
                    if dist_tl < 55.0 and dot > 0.65:
                        s = tl.get_state()
                        if s == carla.TrafficLightState.Red:
                            carla_state = "Red"
                        elif s == carla.TrafficLightState.Yellow:
                            carla_state = "Yellow"   # sadece yavaşlama tetikler, dur değil
            except Exception:
                pass

        if carla_state in ("Red", "Yellow"):
            return carla_state

        # ── YOLO SIGNAL FUSION ───────────────────────────────────────────────
        # Sadece merkezdeki, yeterince büyük, dikey bbox'lara bak
        if frame is not None and signal_boxes:
            best_box   = None
            best_score = 0
            for (x1, y1, x2, y2, cx, h) in signal_boxes:
                in_center = CFG["CENTER_X_MIN"] < cx < CFG["CENTER_X_MAX"]
                if not in_center: continue
                if h < 20:        continue
                if h > best_score:
                    best_score = h
                    best_box   = (x1, y1, x2, y2)

            if best_box and best_score > 30:
                yolo_color = self._analyze_signal_color(frame, *best_box)
                # YOLO kırmızı görüyorsa hafif uyarı (Yellow), sarı görüyorsa yoksay
                # Çünkü YOLO sarı tespiti çok hatalı, gereksiz durmaya yol açıyor
                if yolo_color == "Red": return "Yellow"

        return "Green"

    # =====================================================
    # HELPERS
    # =====================================================

    def _get_speed(self):
        v = self.vehicle.get_velocity()
        return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2) * 3.6

    def _compute_steer(self, ahead=None, dt=0.05):
        """
        Pure Pursuit + Lane Keeping (CTE) Fusion — GÜÇLENDİRİLMİŞ ŞERİT TAKİBİ
        DEĞİŞİKLİKLER:
          - CTE kazancı hıza göre dinamik (CFG parametreli)
          - Waypoint şerit merkezine snap: offset küçükse direkt düzelt
          - CTE düzeltme üst limiti (aşırı salınımı önle)
          - Çoklu waypoint ağırlıklı hedef (smoother heading)
        """
        if ahead is None:
            ahead = CFG["STEER_AHEAD_DIST"]

        speed = self._get_speed()

        dynamic_ahead = np.clip(
            ahead + (speed / 30.0) * 2.0,
            CFG["STEER_LOOKAHEAD_MIN"],
            CFG["STEER_LOOKAHEAD_MAX"]
        )

        v_loc    = self.vehicle.get_location()
        waypoint = self.map.get_waypoint(
            v_loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if waypoint is None:
            return self._steer_smooth

        vt  = self.vehicle.get_transform()
        yaw = math.radians(vt.rotation.yaw)

        # =========================================================
        # LANE KEEPING — CTE (Cross Track Error)  ← GÜÇLENDİRİLDİ
        # =========================================================
        wp_loc = waypoint.transform.location
        dx_wp  = wp_loc.x - vt.location.x
        dy_wp  = wp_loc.y - vt.location.y

        # Araç yerel frame'inde lateral hata (+ = sağa kaymış, - = sola kaymış)
        cte    = -dx_wp * math.sin(yaw) + dy_wp * math.cos(yaw)
        lane_w = waypoint.lane_width if waypoint.lane_width > 0 else 3.5

        # Normalize et ve limitle
        cte_norm = np.clip(cte / (lane_w * 0.5), -1.0, 1.0)

        cte_gain = (
            CFG["CTE_GAIN_LOW_SPD"] if speed < 20.0
            else CFG["CTE_GAIN_HIGH_SPD"]
        )

        cte_correction = float(np.clip(
            cte_gain * cte_norm,
            -CFG["CTE_MAX_CORRECTION"],
            CFG["CTE_MAX_CORRECTION"]
        ))

        # =========================================================
        # PURE PURSUIT — Heading Error  ← AĞIRLIKLI ÇOKLU WAYPOINT
        # =========================================================
        current_wp = waypoint
        waypoints  = []
        total_dist = 0.0

        for _ in range(15):
            next_wps = current_wp.next(2.0)
            if not next_wps:
                break
            next_wp = next_wps[0]
            if next_wp.lane_type != carla.LaneType.Driving:
                break
            waypoints.append(next_wp)
            total_dist += 2.0
            current_wp  = next_wp
            if total_dist >= dynamic_ahead:
                break

        if not waypoints:
            # Waypoint yoksa sadece CTE düzeltmesi
            raw_steer = self.steer_pid.step(cte_correction, dt)
            alpha = CFG["STEER_SMOOTH"]
            if speed < 10.0:   alpha = 0.25
            elif speed > 50.0: alpha = 0.75
            self._steer_smooth = alpha * self._steer_smooth + (1.0 - alpha) * raw_steer
            self._steer_smooth = float(np.clip(self._steer_smooth, -1.0, 1.0))
            return self._steer_smooth

        # Bu şerit merkezini daha erken hissettiriyor
        if len(waypoints) >= 3:
            mid = waypoints[len(waypoints) // 2]
            end = waypoints[-1]
            # %40 orta, %60 son nokta
            target_x = 0.40 * mid.transform.location.x + 0.60 * end.transform.location.x
            target_y = 0.40 * mid.transform.location.y + 0.60 * end.transform.location.y
        else:
            target_x = waypoints[-1].transform.location.x
            target_y = waypoints[-1].transform.location.y

        dx        = target_x - vt.location.x
        dy        = target_y - vt.location.y
        angle_to  = math.atan2(dy, dx)
        heading_err = angle_to - yaw

        while heading_err >  math.pi: heading_err -= 2 * math.pi
        while heading_err < -math.pi: heading_err += 2 * math.pi

        # =========================================================
        # FUSION: Heading + CTE  ← DEĞİŞTİ
        # =========================================================
        # CTE artık bağımsız hesaplanmış ve limitli
        err = heading_err + cte_correction

        raw_steer = self.steer_pid.step(err, dt)

        alpha = CFG["STEER_SMOOTH"]
        if speed < 10.0:   alpha = 0.25
        elif speed > 50.0: alpha = 0.75

        self._steer_smooth = alpha * self._steer_smooth + (1.0 - alpha) * raw_steer
        self._steer_smooth = float(np.clip(self._steer_smooth, -1.0, 1.0))
        return self._steer_smooth

    def _compute_throttle_brake(self, mode, target_spd, speed, dt):
        if mode in ("emergency", "hard_brake"):
            self.speed_pid.reset()
            return 0.0, 1.0

        if mode == "tl_stop":
            self.speed_pid.reset()
            return 0.0, min(1.0, max(0.5, speed / 30.0))

        self.target_speed = 0.65 * self.target_speed + 0.35 * target_spd
        err    = self.target_speed - speed
        output = self.speed_pid.step(err, dt)

        if output > 0:
            throttle = float(np.clip(output, 0.0, CFG["THROTTLE_MAX"]))
            brake    = 0.0
        else:
            throttle = 0.0
            b_raw    = float(np.clip(-output, 0.0, 1.0))
            brake    = b_raw if mode == "hard_brake" else CFG["BRAKE_SMOOTH"] * b_raw

        return throttle, brake

    # =====================================================
    # PARSE RESULTS  ← DEĞİŞTİ
    # =====================================================

    def _parse_results(self, results):
        detections   = {c: 0 for c in VALIDATION_CLASSES}
        box_details  = []
        label_counts = {}
        signal_boxes = []

        if not results or not results[0].boxes:
            return detections, box_details, label_counts, signal_boxes

        img_h = 360  # kamera yüksekliği
        img_w = 640  # kamera genişliği

        for box in results[0].boxes:
            raw_label = self.model.names[int(box.cls[0])].lower()
            label     = LABEL_MAP.get(raw_label, raw_label.upper())

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            h  = y2 - y1
            w  = x2 - x1

            # ── SIGNAL FİLTRESİ ← GÜÇLENDİRİLDİ ────────────────────
            if label == "SIGNAL":
                if w > 3.0 * h:       continue  # çok yatay = tabela
                if h < 18 or w < 10:  continue  # çok küçük = gürültü
                if w > 80 or h > 110: continue  # çok büyük = bina/pano
                if y1 < 15:           continue  # görüntü üst kenarı
                # Trafik lambası h >= w olur (dikey), direk/tabela h < w olur
                if w > 0 and h < w * 0.8:
                    continue
                if y1 > img_h * 0.60:
                    continue
                signal_boxes.append((x1, y1, x2, y2, cx, h))

            if label == "VEHICLE":
                # Boyut filtreleri (eski)
                if h > 250 and w > 300: continue
                if y1 < 5  and h > 200: continue
                if w > 500:             continue
                if h > 200 and w > 250 and (cx < 100 or cx > 540): continue

                # Gerçek araç: h/w genellikle > 0.4 (dik durur)
                # Bina duvarı: h/w çok küçük (yatay, geniş)
                if w > 0 and (h / w) < 0.30 and h > 80:
                    continue  # Çok yatay → bina/duvar yüzeyi

                if y1 < img_h * 0.25 and h > 150:
                    continue

                in_center = CFG["CENTER_X_MIN"] < cx < CFG["CENTER_X_MAX"]
                if not in_center and h > 180 and w > 200:
                    continue

            if label == "PERSON":
                if y2 < img_h * 0.20:
                    continue
                if w > 120 and (h / w) < 0.50:
                    continue

            if label == "OBSTACLE" and w > 300 and h < 200:
                continue

            if label in VALIDATION_CLASSES:
                detections[label] += 1

            box_details.append((label, cx, h))
            label_counts[label] = label_counts.get(label, 0) + 1

        return detections, box_details, label_counts, signal_boxes

    def _cleanup_actors(self):
        self.actors = [a for a in self.actors if a is not None and a.is_alive]

    def _draw_hud(self, annotated, frame_raw):
        small = cv2.resize(frame_raw, (180, 100))
        annotated[10:110, 10:190] = small
        cv2.rectangle(annotated, (10, 10), (190, 110), (255, 255, 255), 2)
        cv2.putText(annotated, "FOGGY CAMERA",
                    (22, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        risk_color = (
            (0, 255, 0)   if self.risk_score < 0.25 else
            (0, 200, 255) if self.risk_score < 0.55 else
            (0, 100, 255) if self.risk_score < 0.80 else
            (0, 0, 255)
        )
        bar_w = int(self.risk_score * 195)
        cv2.rectangle(annotated, (10, 115), (205, 130), (40, 40, 40), -1)
        cv2.rectangle(annotated, (10, 115), (10 + bar_w, 130), risk_color, -1)
        cv2.putText(annotated, f"RISK {self.risk_score:.2f}",
                    (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.42, risk_color, 1)

        overlay = annotated.copy()
        cv2.rectangle(overlay, (430, 8), (637, 230), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0, annotated)

        lines = [
            (f"SPEED   {int(self._get_speed())} km/h", (0, 255, 80)),
            (f"TARGET  {int(self.target_speed)} km/h",  (180, 255, 100)),
            ("DEHAZE  ON",                               (0, 255, 255)),
            (f"MODE    {self.current_mode.upper()}",
             (0, 200, 255) if self.current_mode == "go" else (0, 80, 255)),
            (f"TL      {self.tl_state}",
             (0, 255, 0) if self.tl_state == "Green" else (0, 80, 255)),
            (f"DIST    {self.dist:.1f} m",              (200, 200, 200)),
            (f"TTC     {self.ttc:.1f} s",               (0, 140, 255)),
            (f"PREC    %{int(self.validation.mean_precision)}", (255, 255, 255)),
            (f"RECALL  %{int(self.validation.mean_recall)}",    (220, 220, 220)),
            (f"ACC     %{int(self.validation.mean_accuracy)}",  (0, 200, 255)),
        ]
        for i, (txt, col) in enumerate(lines):
            cv2.putText(annotated, txt,
                        (440, 30 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)

        return annotated

    # =====================================================
    # MAIN LOOP
    # =====================================================

    def run(self):
        self.setup_sensors()
        self.spawn_world_objects()

        cv2.namedWindow("CARLA AI DEHAZE SYSTEM", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CARLA AI DEHAZE SYSTEM", CFG["DISPLAY_W"], CFG["DISPLAY_H"])

        try:
            while True:
                try:
                    self.world.tick()
                except Exception as e:
                    print(f"[WARN] tick hatası: {e}")
                    continue

                if not self.vehicle.is_alive:
                    print("[WARN] Araç yok edildi, çıkılıyor.")
                    break

                if not self.cam_queue:
                    continue

                image = self.cam_queue.pop()
                frame = np.frombuffer(
                    image.raw_data, dtype=np.uint8
                ).reshape((360, 640, 4))[:, :, :3]

                run_dehaze = (self.frame_count % CFG["DEHAZE_SKIP"] == 0)
                if run_dehaze:
                    self._last_dehazed = self.dehaze_frame(frame)
                dehazed = self._last_dehazed if self._last_dehazed is not None else frame

                curr_time      = time.time()
                dt             = max(curr_time - self.prev_time, 0.001)
                self.prev_time = curr_time

                self.dist, self.ttc = self.lidar_proc.update(
                    self.lidar_points, dt, self.vehicle, self.map
                )

                run_yolo = (self.frame_count % CFG["YOLO_SKIP"] == 0)
                if run_yolo:
                    results = self.model.track(
                        dehazed, imgsz=640, persist=True,
                        device=self.device, verbose=False,
                        conf=0.15, iou=0.45,
                    )
                    (self._last_detections,
                     self._last_box_details,
                     self._last_label_counts,
                     self._last_signal_boxes) = self._parse_results(results)
                    self._last_results = results
                else:
                    results = self._last_results

                detections  = self._last_detections
                box_details = self._last_box_details

                self.tl_state   = self._check_traffic_light(dehazed, self._last_signal_boxes)
                speed           = self._get_speed()
                self.risk_score = self.risk_scorer.compute(
                    self.dist, self.ttc, box_details, detections, self.tl_state,
                )

                self.current_mode, target_spd = self.decision.decide(
                    self.risk_score, self.dist, self.ttc,
                    speed, box_details, self.tl_state,
                )

                self.throttle, self.brake = self._compute_throttle_brake(
                    self.current_mode, target_spd, speed, dt,
                )

                ahead = (
                    10.0 if self.current_mode == "go" else
                    6.0 if self.current_mode in ("caution", "soft_brake") else
                    5.0
                )

                self.steer = self._compute_steer(ahead, dt)

                self.vehicle.apply_control(carla.VehicleControl(
                    throttle=float(self.throttle),
                    brake=float(self.brake),
                    steer=float(self.steer),
                ))

                if self.frame_count % CFG["VALIDATION_SKIP"] == 0:
                    gt = self.validation.get_gt(self.vehicle, self.world)
                    self.validation.update(detections, gt)

                if self.frame_count % CFG["CLEANUP_INTERVAL"] == 0:
                    self._cleanup_actors()

                self.frame_count += 1
                self.telemetry.record(
                    self.frame_count, speed,
                    self.risk_score, self.ttc, self.dist,
                    self.current_mode, self.tl_state,
                    self.throttle, self.brake, self.steer,
                )

                if results is not None:
                    annotated = results[0].plot()
                else:
                    annotated = dehazed.copy()
                annotated = self._draw_hud(annotated, frame)

                cv2.imshow("CARLA AI DEHAZE SYSTEM", annotated)
                if cv2.waitKey(1) == ord("q"):
                    break

        finally:
            print("[CLEANING ACTORS]")
            for actor in reversed(self.actors):
                try:
                    actor.destroy()
                except:
                    pass

            self.validation.final_report()
            self.telemetry.summary()
            self.validation.save_chart("validation_report.png")
            cv2.destroyAllWindows()


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    agent = AutonomousAgent("D:/OtonomProje/best.pt")
    agent.run()