import carla
import cv2
import numpy as np
from ultralytics import YOLO
import sys

def main():
    print("--- Program Başlatılıyor ---")
    
    try:
        # 1. Bağlantı Denemesi
        print("CARLA Sunucusuna bağlanılıyor (Port: 2000)...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0) # 10 saniye bekle
        world = client.get_world()
        print("CARLA Bağlantısı Başarılı!")

        # 2. Model Yükleme
        print("YOLOv11 Modeli yükleniyor...")
        model = YOLO('yolo11n.pt') 
        print("Model Başarıyla Yüklendi!")

        # 3. Araç Oluşturma
        print("Araç oluşturuluyor...")
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('model3')[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(True)
        print("Araç oluşturuldu ve otopilota alındı.")

        # 4. Kamera Sensörü
        print("Kamera sensörü yerleştiriliyor...")
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

        def process_image(image):
            i = np.frombuffer(image.raw_data, dtype=np.uint8)
            i2 = i.reshape((image.height, image.width, 4))
            frame = i2[:, :, :3]
            
            results = model(frame, conf=0.4, verbose=False)
            annotated_frame = results[0].plot()
            
            cv2.imshow("CARLA YOLOv11 Test", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

        camera.listen(lambda data: process_image(data))
        print("\n>>> SİSTEM ÇALIŞIYOR. Görüntü penceresini kontrol edin.")
        print(">>> Durdurmak için CMD ekranında Ctrl+C tuşlarına basın.")

        while True:
            world.wait_for_tick()

    except Exception as e:
        print(f"\nHATA OLUŞTU: {e}")
    finally:
        print("\nTemizlik yapılıyor ve kapatılıyor...")

if __name__ == '__main__':
    main()