# veri_hazirlama.py (GÜNCEL - Dinamik Boyutlu)

import torch
import kagglehub
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Fonksiyona 'image_size' parametresi ekliyoruz.
# Varsayılan 224, böylece eski kodların bozulmadan çalışır.
def get_dataloaders(batch_size=64, image_size=224):
    """
    Veri setini indirir ve istenen boyutta (image_size) DataLoader'ları döndürür.
    InceptionV3 için image_size=299, diğerleri için 224 kullanılmalı.
    """

    # Transformları dinamik boyuta göre ayarlıyoruz
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),  # <-- Burası artık dinamik
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),  # <-- Burası artık dinamik
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    print(f"KaggleHub'dan veri seti yolu alınıyor... (Hedef Boyut: {image_size}x{image_size})")
    # Veri setini indir bul ve yolunu al
    # (Kagglehub zaten indirdiyse tekrar indirmez, yolu verir)
    dataset_path = kagglehub.dataset_download("msambare/fer2013")

    # train ve test yollarını belirliyoruz
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')

    print(f"Eğitim verisi buradan okunacak: {train_dir}")
    print(f"Test verisi buradan okunacak: {test_dir}")

    # 1. Dataset nesnelerini oluştur
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=test_transform
    )

    print(f"Eğitim verisi yüklendi. Toplam: {len(train_dataset)} örnek.")
    print(f"Test verisi yüklendi. Toplam: {len(test_dataset)} örnek.")

    # 2. DataLoader nesnelerini oluştur
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4  # Veri yüklemeyi hızlandırır
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # 3. Sınıf sayısını al
    num_classes = len(train_dataset.classes)
    print(f"Bulunan sınıflar: {train_dataset.classes}")
    print(f"Toplam sınıf sayısı: {num_classes}")

    return train_loader, test_loader, num_classes