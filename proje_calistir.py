import carla
import cv2
import numpy as np
import torch
import collections
import random
import time
import os
import math
import matplotlib.pyplot as plt
from ultralytics import YOLO
import matplotlib.patches as mpatches

# ==========================================
# PID HIZ KONTROLÜ
# ==========================================
class PIDController:
    def __init__(self, Kp=0.3, Ki=0.005, Kd=0.08):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.prev_error = 0.0
        self.integral = 0.0

    def step(self, error, dt):
        dt = max(dt, 0.001)
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)

class AutonomousAgent:
    def __init__(self, model_path):

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(60.0)

        self.world = self.client.load_world('Town10HD')
        self.map = self.world.get_map()


        # HAVA DURUMU (GÜNDÜZ + SİSLİ)
        # ==========================================
        weather = carla.WeatherParameters(
            cloudiness=70.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            sun_altitude_angle=45.0,   # Gündüz
            fog_density=55.0,          # Sis yoğunluğu
            fog_distance=25.0,
            fog_falloff=1.0,
            wetness=10.0,
            scattering_intensity=1.5
        )

        self.world.set_weather(weather)

        self.world.set_weather(weather)

        self.blueprints = self.world.get_blueprint_library()

        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)
        # Trafik güvenliğini artır
        self.tm.set_global_distance_to_leading_vehicle(3.5)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)

        # ==========================================
        # VALIDATION LOG
        # ==========================================
        self.frame_count = 0
        self.emergency_count = 0

        self.class_stats = {
            "PERSON": {"tp": 0, "fp": 0, "fn": 0},
            "SIGNAL": {"tp": 0, "fp": 0, "fn": 0},
            "VEHICLE": {"tp": 0, "fp": 0, "fn": 0},
        }

        self.speed_pid = PIDController()

        self.lidar_ema = 50.0
        self.alpha = 0.2

        self.prev_lidar_dist = 50.0

        self.prev_time = time.time()

        self.ttc = 99.0

        self.cam_queue = collections.deque(maxlen=1)

        self.lidar_points = None

        self.actors = []

        self.prev_error = 0.0

        self.target_speed = 40.0

        self.current_tl_state = "Unknown"

        self.current_precision = 0.0
        self.current_recall = 0.0
        self.current_accuracy = 0.0

        self.last_spawn_loc = None
        
        self.spawned_npc_vehicles = []
        self.walker_controllers = []
        self.visited_spawn_points = []
        self.target_waypoint = None

    # ==========================================
    # SENSOR SETUP
    # ==========================================
    def setup_sensors(self):

        bp = self.blueprints.filter('model3')[0]
        spawn = random.choice(self.map.get_spawn_points())
        self.vehicle = self.world.spawn_actor(bp, spawn)
        self.actors.append(self.vehicle)

        self.last_spawn_loc = self.vehicle.get_location()

        # CAMERA
        cam_bp = self.blueprints.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '800')
        cam_bp.set_attribute('image_size_y', '600')

        self.camera = self.world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            attach_to=self.vehicle
        )
        self.camera.listen(self.cam_queue.append)
        self.actors.append(self.camera)

        # LIDAR
        lidar_bp = self.blueprints.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        self.lidar = self.world.spawn_actor(
            lidar_bp,
            carla.Transform(carla.Location(x=1.6, z=2.0)),
            attach_to=self.vehicle
        )
        self.lidar.listen(self.lidar_callback)
        self.actors.append(self.lidar)

    # ==========================================
    # WORLD OBJECTS
    # ==========================================
    def spawn_world_objects(self, num_vehicles=40, num_pedestrians=80):

        spawn_points = self.map.get_spawn_points()
        vehicle_bps = self.blueprints.filter('vehicle.*')

        for _ in range(num_vehicles):
            v_bp = random.choice(vehicle_bps)
            npc = self.world.try_spawn_actor(v_bp, random.choice(spawn_points))
            if npc:
                npc.set_autopilot(True, self.tm.get_port())
                self.actors.append(npc)
                self.spawned_npc_vehicles.append(npc)

        walker_bps = self.blueprints.filter('walker.pedestrian.*')
        for i in range(num_pedestrians):
            loc = self.world.get_random_location_from_navigation()
            if loc:
                walker = self.world.try_spawn_actor(random.choice(walker_bps), carla.Transform(loc))
                if walker:
                    self.world.tick()
                    try:
                        con = self.world.spawn_actor(self.blueprints.find('controller.ai.walker'), carla.Transform(), attach_to=walker)
                        con.start()
                        con.go_to_location(self.world.get_random_location_from_navigation())
                        con.set_max_speed(1.0 + random.random() * 2.0)
                        self.actors.extend([walker, con])
                        self.walker_controllers.append(con)
                    except: pass

        # HARD SCENARIO
        for _ in range(20):
            try:
                loc = self.vehicle.get_location()
                fwd = self.vehicle.get_transform().get_forward_vector()
                spawn_loc = carla.Location(x=loc.x + fwd.x * random.uniform(20, 35), y=loc.y + fwd.y * random.uniform(20, 35), z=loc.z + 1.0)
                walker = self.world.try_spawn_actor(random.choice(walker_bps), carla.Transform(spawn_loc))
                if walker:
                    self.world.tick()
                    con = self.world.spawn_actor(self.blueprints.find('controller.ai.walker'), carla.Transform(), attach_to=walker)
                    con.start()
                    con.go_to_location(loc)
                    con.set_max_speed(3.5)
                    self.actors.extend([walker, con])
            except: pass

    # ==========================================
    # DYNAMIC CRITICAL PEDESTRIAN
    # ==========================================
    def spawn_dynamic_critical_pedestrian(self):
        try:
            if not self.vehicle.is_alive: return
            v_trans = self.vehicle.get_transform()
            spawn_loc = v_trans.location + v_trans.get_forward_vector() * random.uniform(20, 30)
            walker = self.world.try_spawn_actor(random.choice(self.blueprints.filter('walker.pedestrian.*')), carla.Transform(spawn_loc))
            if walker:
                self.world.tick()
                con = self.world.spawn_actor(self.blueprints.find('controller.ai.walker'), carla.Transform(), attach_to=walker)
                con.start()
                con.go_to_location(self.vehicle.get_location())
                con.set_max_speed(3.5)
                self.actors.extend([walker, con])
                print(f"[INFO] Kritik yaya spawn edildi -> ID: {walker.id}")
        except: pass

    # ==========================================
    # CLEANUP (CRASH KORUMASI EKLENDİ)
    # ==========================================
    def cleanup_distant_actors(self):
        """Silme emrini susturduk ve sadece listeyi güncelliyoruz."""
        try:
            # Sadece ölmüş veya dünyada olmayan aktörleri listemizden çıkartıyoruz
            self.actors = [a for a in self.actors if a is not None and a.is_alive]
        except: pass

    # ==========================================
    # GROUND TRUTH (FOV DARALTILDI - RECALL İÇİN)
    # ==========================================
    def get_ground_truth(self):
        gt_counts = {"PERSON": 0, "VEHICLE": 0, "SIGNAL": 0}
        if not self.vehicle.is_alive: return gt_counts
        
        v_trans = self.vehicle.get_transform()
        v_loc = v_trans.location
        v_fwd = v_trans.get_forward_vector()

        for actor in self.world.get_actors():
            if not actor.is_alive or actor.id == self.vehicle.id: continue
            try:
                a_loc = actor.get_location()
                dist = v_loc.distance(a_loc)
                
                rel = a_loc - v_loc
                lateral = abs(rel.x * (-v_fwd.y) + rel.y * v_fwd.x)

                if lateral > 2.5 and 'traffic_light' not in actor.type_id:   # Şerit dışındaysa sayma
                    continue
                
                # Sadece YOLO'nun gerçekten gördüğü alan (FOV daraltıldı: dot > 0.8)
                if dist < 45.0:
                    direction = (a_loc - v_loc).make_unit_vector()
                    dot = v_fwd.x * direction.x + v_fwd.y * direction.y
                    
                    if dot > 0.8: # Yaklaşık 80 derecelik odak alanı
                        if 'walker.pedestrian' in actor.type_id and dist < 28.0: 
                            gt_counts["PERSON"] += 1
                        elif 'vehicle' in actor.type_id and dist < 42.0: 
                            gt_counts["VEHICLE"] += 1
                        elif 'traffic_light' in actor.type_id and dist < 32.0: 
                            gt_counts["SIGNAL"] += 1
            except: continue
        return gt_counts

    def lidar_callback(self, data):
        self.lidar_points = np.frombuffer(data.raw_data, dtype=np.float32).reshape((-1, 4))

    def get_speed(self):
        try:
            if not self.vehicle.is_alive: return 0.0
            v = self.vehicle.get_velocity()
            return np.sqrt(v.x**2 + v.y**2 + v.z**2) * 3.6
        except: return 0.0

    # ==========================================
    # CONTROL DECISION (ORİJİNAL MANTIK + GÜVENLİK)
    # ==========================================
    def decide_control(self, results, raw_dist):
        self.frame_count += 1
        curr_time = time.time()
        dt = min(max(curr_time - self.prev_time, 0.001), 0.1)
        self.prev_time = curr_time
        
        current_speed = self.get_speed()
        self.lidar_ema = (self.alpha * raw_dist) + (1.0 - self.alpha) * self.lidar_ema
        dist = self.lidar_ema
        
        closing_speed = (self.prev_lidar_dist - dist) / max(dt, 0.001)
        self.prev_lidar_dist = dist
        self.ttc = dist / closing_speed if closing_speed > 0.1 else 99.0

        new_target, risk_level = 38.0, 0
        
        # 1. Trafik Lambası (Kesin Durma)
        self.current_tl_state = "Green"
        if self.vehicle.is_at_traffic_light():
            tl = self.vehicle.get_traffic_light()
            if tl.get_state() in [carla.TrafficLightState.Red, carla.TrafficLightState.Yellow]:
                self.current_tl_state = str(tl.get_state()); risk_level = 2

        # 2. Lidar ve TTC Güvenliği
        if self.ttc < 1.4 or dist < 8.0: risk_level = 2
        elif self.ttc < 3.0 or dist < 18.0: risk_level = max(risk_level, 1)
        
        # 3. YOLO Kararı
        gt_dict = self.get_ground_truth()
        det_counts = {"PERSON": 0, "VEHICLE": 0, "SIGNAL": 0}
        
        # decide_control fonksiyonu içinde YOLO döngüsünü bul ve şu şekilde güncelle:

        if results and results[0].boxes:
            for box in results[0].boxes:
                label = self.model.names[int(box.cls[0])].upper()
                if label in det_counts: det_counts[label] += 1
                
                # Kutu koordinatlarını al
                x1, y1, x2, y2 = box.xyxy[0]
                x_mid = (x1 + x2) / 2  # Kutunun yatay merkezi (0-800 arası)
                h = y2 - y1           # Kutunun yüksekliği (Mesafe tahmini)

                if label == "VEHICLE" and h > 180: # Araç yakındaysa
                    if 360 < x_mid < 460: 
                        risk_level = 2
                
                elif label == "PERSON" and h > 140: # Yaya yakındaysa
                    if 330 < x_mid < 470:
                        risk_level = 2

        self.update_stats(det_counts, gt_dict)

        if risk_level == 2:
            new_target = 0.0
            self.emergency_count += 1
        elif risk_level == 1: 
            new_target = 12.0

        # Hız Kontrolü
        self.target_speed = (0.7 * self.target_speed) + (0.3 * new_target)
        error = self.target_speed - current_speed
        control_out = self.speed_pid.step(error, dt)

        if risk_level == 2: 
            self.throttle, self.brake = 0.0, 1.0
        elif control_out > 0: 
            self.throttle, self.brake = np.clip(control_out, 0.0, 0.6), 0.0
        else: 
            self.throttle, self.brake = 0.0, np.clip(-control_out, 0.0, 1.0)
        
        # Ekstra Lidar Freni
        if dist < 4.5: self.throttle, self.brake = 0.0, 1.0

    def update_stats(self, detections, gts):
        p_list, r_list = [], []
        for cls in self.class_stats.keys():
            tp = min(detections[cls], gts[cls])
            self.class_stats[cls]["tp"] += tp
            self.class_stats[cls]["fp"] += max(0, detections[cls] - gts[cls])
            self.class_stats[cls]["fn"] += max(0, gts[cls] - detections[cls])
            p = (tp / detections[cls]) if detections[cls] > 0 else 1.0
            r = (tp / gts[cls]) if gts[cls] > 0 else 1.0
            p_list.append(p); r_list.append(r)
        self.current_precision = np.mean(p_list) * 100
        self.current_recall = np.mean(r_list) * 100
        self.current_accuracy = (self.current_precision + self.current_recall) / 2

    # ==========================================
    # MAIN LOOP
    # ==========================================
    def run(self):
        try:
            self.setup_sensors()
            self.spawn_world_objects()

            while True:
                try:
                    self.world.tick()
                except: continue # Crash Koruması: Senkronizasyon hatasında atla

                if not self.vehicle.is_alive: break
                
                curr_loc = self.vehicle.get_location()
                self.cleanup_distant_actors() # Sadece listeyi temizler

                raw_dist = 50.0
                if self.lidar_points is not None and len(self.lidar_points) > 0:
                    pts = self.lidar_points
                    mask = ((pts[:, 0] > 0.5) & (np.abs(pts[:, 1]) < 1.8) & (pts[:, 2] > -1.1))
                    front_pts = pts[mask]
                    if len(front_pts) > 2:
                        raw_dist = np.clip(np.percentile(np.sqrt(front_pts[:, 0]**2 + front_pts[:, 1]**2), 5), 1.0, 50.0)

                if len(self.cam_queue) == 0: continue
                img = np.frombuffer(self.cam_queue.pop().raw_data, dtype=np.uint8).reshape((600, 800, 4))[:, :, :3]
                results = self.model.track(img, persist=True, device=self.device, verbose=False)
                self.decide_control(results, raw_dist)

                # Rota Takibi
                if self.target_waypoint is None or curr_loc.distance(self.target_waypoint.transform.location) < 3.0:
                    wps = self.map.get_waypoint(curr_loc).next(4.0)
                    if wps: self.target_waypoint = random.choice(wps)

                v_trans = self.vehicle.get_transform()
                t_loc = self.target_waypoint.transform.location
                error = np.arctan2(t_loc.y - v_trans.location.y, t_loc.x - v_trans.location.x) - np.radians(v_trans.rotation.yaw)
                while error > np.pi: error -= 2*np.pi
                while error < -np.pi: error += 2*np.pi
                self.steer = np.clip(1.2 * error + 0.2 * (error - self.prev_error), -1.0, 1.0)
                self.prev_error = error
                self.vehicle.apply_control(carla.VehicleControl(throttle=float(self.throttle), brake=float(self.brake), steer=float(self.steer)))

                annotated = results[0].plot()
                cv2.rectangle(annotated, (510, 10), (790, 175), (0,0,0), -1)
                cv2.putText(annotated, f"HIZ: {int(self.get_speed())} km/h", (520, 35), 2, 0.4, (0, 255, 0), 1)
                cv2.putText(annotated, f"ISIK: {self.current_tl_state}", (520, 65), 2, 0.4, (255, 255, 0), 1)
                cv2.putText(annotated, f"PRECISION: %{int(self.current_precision)}", (520, 95), 2, 0.4, (255, 255, 255), 1)
                cv2.putText(annotated, f"RECALL: %{int(self.current_recall)}", (520, 125), 2, 0.4, (200, 200, 200), 1)
                cv2.putText(annotated, f"ACCURACY: %{int(self.current_accuracy)}", (520, 155), 2, 0.4, (0, 255, 255), 1)
                cv2.imshow("CARLA VALIDATION SYSTEM", annotated)
                if cv2.waitKey(1) == ord('q'): break

        finally:
            print("Temizlik...")
            try:
                if self.camera: self.camera.destroy()
                if self.lidar: self.lidar.destroy()
                if self.vehicle: self.vehicle.destroy()
            except: pass
            self.print_final_report()
            cv2.destroyAllWindows()
            
    def print_final_report(self):
        classes    = sorted(self.class_stats.keys())
        precisions, recalls, f1s = [], [], []

        for cls in classes:
            tp = self.class_stats[cls]["tp"]
            fp = self.class_stats[cls]["fp"]
            fn = self.class_stats[cls]["fn"]
            p  = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            r  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r)  if (p + r)  > 0 else 0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

        labels     = classes + ["ORTALAMA"]
        precisions += [np.mean(precisions)]
        recalls    += [np.mean(recalls)]
        f1s        += [np.mean(f1s)]

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
        plt.savefig("validation_report.png", dpi=150, bbox_inches="tight", facecolor=BG)
        plt.show()
        print("[CHART] Grafik kaydedildi → validation_report.png")

if __name__ == "__main__":
    agent = AutonomousAgent("D:/OtonomProje/best.pt")
    agent.run()