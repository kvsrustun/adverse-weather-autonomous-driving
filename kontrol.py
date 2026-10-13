import torch

# Kendi yoluna göre güncelle
gan_path = "D:/OtonomProje/pytorch-CycleGAN-and-pix2pix/checkpoints/carla_pix2pix/latest_net_G.pth"

checkpoint = torch.load(gan_path, map_location='cpu')
print("--- DOSYADAKİ KATMAN İSİMLERİ ---")
for key in list(checkpoint.keys())[:20]: 
    print(key)