import carla
import cv2
import numpy as np
import os
import random
import time

# ============= AYARLAR =============
SAVE_PATH = r"D:\OtonomProje\carla_gan_data"
IMG_WIDTH = 800
IMG_HEIGHT = 600
CAPTURE_INTERVAL = 30    # Her X karede bir kayıt (Yeterli mesafe için)
TOTAL_SAMPLES = 3000     
TELEPORT_EVERY = 50      # Mekan döngüsünü kırmak için zorunlu ışınlanma
STUCK_THRESHOLD = 0.5    # Araç sıkıştı kontrolü

# Klasörleri oluştur
for sub in ["train_A", "train_B"]:
    os.makedirs(os.path.join(SAVE_PATH, sub), exist_ok=True)

class DataCollector:
    def __init__(self):
        # CARLA Bağlantısı
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)
        
        # Dünyayı al (Öneri: CARLA ekranında Town10HD veya Town03 yüklü olmalı)
        self.world = self.client.get_world() 
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.tm = self.client.get_trafficmanager(8000)
        
        # Senkron Mod Ayarları
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(self.settings)
        self.tm.set_synchronous_mode(True)

        self.vehicle = None
        self.camera = None
        self.current_image = None
        self.count = 0
        self.last_location = None
        self.actors = [] # NPC'leri temizlemek için liste

        # Hava Durumu Tanımları
        self.foggy_weather = carla.WeatherParameters(
            cloudiness=100.0, precipitation=40.0, wetness=100.0, 
            fog_density=80.0, fog_distance=5.0, sun_altitude_angle=20.0
        )
        self.clear_weather = carla.WeatherParameters.ClearNoon

    def spawn_npcs(self, n_vehicles=40, n_walkers=60):
        """Şehri diğer araçlarla ve yayalarla doldurur."""
        print(f"[NPC] {n_vehicles} araç ve {n_walkers} yaya oluşturuluyor...")
        
        # Araçları oluştur
        v_blueprints = self.blueprint_library.filter('vehicle.*')
        spawn_points = self.map.get_spawn_points()
        
        for _ in range(n_vehicles):
            bp = random.choice(v_blueprints)
            v = self.world.try_spawn_actor(bp, random.choice(spawn_points))
            if v:
                v.set_autopilot(True, self.tm.get_port())
                self.actors.append(v)

        # Yayaları oluştur
        walker_bps = self.blueprint_library.filter('walker.pedestrian.*')
        for _ in range(n_walkers):
            spawn_loc = self.world.get_random_location_from_navigation()
            if spawn_loc:
                w_bp = random.choice(walker_bps)
                walker = self.world.try_spawn_actor(w_bp, carla.Transform(spawn_loc))
                if walker:
                    self.world.tick()
                    # Yayayı yürütmek için kontrolcü ekle
                    con_bp = self.blueprint_library.find('controller.ai.walker')
                    con = self.world.spawn_actor(con_bp, carla.Transform(), attach_to=walker)
                    con.start()
                    con.go_to_location(self.world.get_random_location_from_navigation())
                    con.set_max_speed(1.5) # İnsan hızı
                    self.actors.extend([walker, con])

    def camera_callback(self, image):
        img = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((IMG_HEIGHT, IMG_WIDTH, 4))
        self.current_image = img[:, :, :3]

    def setup_ego_vehicle(self):
        """Kendi aracımızı ve kameramızı kurar."""
        spawn_point = random.choice(self.map.get_spawn_points())
        bp = self.blueprint_library.filter('model3')[0]
        self.vehicle = self.world.spawn_actor(bp, spawn_point)
        self.actors.append(self.vehicle)

        cam_bp = self.blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(IMG_WIDTH))
        cam_bp.set_attribute('image_size_y', str(IMG_HEIGHT))
        self.camera = self.world.spawn_actor(
            cam_bp, carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=self.vehicle)
        self.camera.listen(lambda image: self.camera_callback(image))
        self.actors.append(self.camera)

        # Otopilot ayarları
        self.vehicle.set_autopilot(True, self.tm.get_port())
        self.tm.ignore_lights_percentage(self.vehicle, 100.0) # Durmamak için ışıkları boşver
        self.tm.random_left_lanechange_percentage(self.vehicle, 50.0)
        self.tm.random_right_lanechange_percentage(self.vehicle, 50.0)

    def teleport_ego(self):
        """Aracı haritanın rastgele başka bir yerine fırlatır."""
        new_pos = random.choice(self.map.get_spawn_points())
        self.vehicle.set_transform(new_pos)
        self.last_location = new_pos.location
        # Görüntünün oturması ve otopilotun kendine gelmesi için bekle
        for _ in range(10): self.world.tick()

    def run(self):
        try:
            self.spawn_npcs()
            self.setup_ego_vehicle()
            print("Hazırlık tamamlandı. Veri toplama başlıyor...")

            stuck_counter = 0
            while self.count < TOTAL_SAMPLES:
                
                # 1. Her 50 karede bir zorla yer değiştir (Mekan çeşitliliği)
                if self.count > 0 and self.count % TELEPORT_EVERY == 0:
                    print(f"\n[INFO] {self.count}. kare: Başka bir bölgeye ışınlanıyor...")
                    self.teleport_ego()

                # 2. CARLA'yı ilerlet
                for _ in range(CAPTURE_INTERVAL):
                    self.world.tick()

                # Sıkışma Kontrolü
                curr_loc = self.vehicle.get_location()
                dist = curr_loc.distance(self.last_location) if self.last_location else 10.0
                if dist < STUCK_THRESHOLD: stuck_counter += 1
                else: stuck_counter = 0
                self.last_location = curr_loc

                if stuck_counter >= 3:
                    print("[UYARI] Araç trafikte takıldı. Işınlanıyor...")
                    self.teleport_ego()
                    stuck_counter = 0
                    continue

                # 3. VERİ KAYIT (Garantili Eşleşmiş Çift)
                # A: Sisli (Foggy)
                self.world.set_weather(self.foggy_weather)
                for _ in range(3): self.world.tick() # Weather render süresi
                if self.current_image is not None:
                    cv2.imwrite(f"{SAVE_PATH}/train_A/img_{self.count}.png", self.current_image)

                # B: Temiz (Clear)
                self.world.set_weather(self.clear_weather)
                for _ in range(3): self.world.tick()
                if self.current_image is not None:
                    cv2.imwrite(f"{SAVE_PATH}/train_B/img_{self.count}.png", self.current_image)

                self.count += 1
                if self.count % 10 == 0:
                    print(f"İlerleme: {self.count}/{TOTAL_SAMPLES} | Kalan: {TOTAL_SAMPLES - self.count}")

        except KeyboardInterrupt:
            print("\nKullanıcı durdurdu.")
        finally:
            print("Temizleniyor...")
            self.settings.synchronous_mode = False
            self.world.apply_settings(self.settings)
            # Tüm NPC ve aktörleri sil
            for actor in self.actors:
                if actor is not None:
                    actor.destroy()
            print("Bitti.")

if __name__ == "__main__":
    collector = DataCollector()
    collector.run()