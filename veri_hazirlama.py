# veri_hazirlama.py (Yeni Sürüm)

import torch
import kagglehub
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Transformları Tanımlama
# SOTA modeller (ResNet, VGG) genellikle 3 kanallı (RGB) girdi bekler.
# transforms.Grayscale ile gri görüntüyü 3 kanala kopyalayacağız.
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # SOTA modeller için
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    #transforms.RandomCrop(48, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 3 kanal için
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # SOTA modeller için
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 3 kanal için
])


# 2. Ana Fonksiyon
def get_dataloaders(batch_size=64):
    """
    Veri setini indirir, hazırlar ve DataLoader'ları döndürür.
    ImageFolder kullanarak train/test klasörlerinden okur.
    """
    print("KaggleHub'dan veri seti yolu alınıyor...")
    # Veri setini indir bul ve yolunu al
    dataset_path = kagglehub.dataset_download("msambare/fer2013")

    # train ve test yollarını belirliyoruz
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')

    print(f"Eğitim verisi buradan okunacak: {train_dir}")
    print(f"Test verisi buradan okunacak: {test_dir}")

    # 1. Dataset nesnelerini oluştur
    # ImageFolder, alt klasörleri otomatik olarak sınıf olarak tanır
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

    # 3. Sınıf sayısını al (ImageFolder bunu otomatik sağlar)
    num_classes = len(train_dataset.classes)
    print(f"Bulunan sınıflar: {train_dataset.classes}")
    print(f"Toplam sınıf sayısı: {num_classes}")

    return train_loader, test_loader, num_classes