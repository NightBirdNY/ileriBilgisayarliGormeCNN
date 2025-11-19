import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from veri_hazirlama import get_dataloaders  # Test verisini almak için
from modeller import get_model  # SOTA modelleri yüklemek için
from telos_model import teloCNN  # Kendi modelini yüklemek için
import os

# --- Ayarlar ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
MODEL_PATH = "saved_models"

# Model listesi (egitim.py ile aynı olmalı)
model_listesi = ["resnet50", "alexnet", "vgg16", "vgg19", "densenet121", "efficientnet_b0", "inception_v3", "teloCNN"]

# --- Veriyi Yükle ---
# Analiz için veri artırma (augmentation) KULLANILMAYAN test loader'a ihtiyacımız var
# veri_hazirlama.py'daki test_transform'u sadeleştirebiliriz
# Ama şimdilik mevcut olanı kullanalım:
_, test_loader, NUM_CLASSES = get_dataloaders(batch_size=BATCH_SIZE)
class_names = test_loader.dataset.classes  # Sınıf isimlerini al ('angry', 'happy' vb.)


# --- Yardımcı Fonksiyon: Tahminleri Topla ---
def get_predictions(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)


# --- Ana Analiz Döngüsü ---
for model_adi in model_listesi:
    print(f"\n--- Analiz Ediliyor: {model_adi} ---")

    # --- BOYUT AYARI ---
    if model_adi == "inception_v3":
        hedef_boyut = 299
    else:
        hedef_boyut = 224

    # get_dataloaders'ı doğru boyutla çağırıyoruz
    # Sadece test loader lazım
    _, test_loader, NUM_CLASSES = get_dataloaders(batch_size=BATCH_SIZE, image_size=hedef_boyut)

    # Modeli yeniden oluştur ve Yükle
    if model_adi == "teloCNN":
        model = teloCNN(num_classes=NUM_CLASSES)
    else:
        model = get_model(model_adi, num_classes=NUM_CLASSES)

    # Kayıtlı ağırlıkları yükle
    model_save_path = f"{MODEL_PATH}/{model_adi}.pth"
    try:
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    except FileNotFoundError:
        print(f"HATA: {model_save_path} bulunamadı. Lütfen önce modeli eğitin.")
        continue

    model = model.to(DEVICE)

    # 1. Tüm tahminleri al
    y_pred, y_true = get_predictions(model, test_loader)

    # 2. Sınıflandırma Raporu (Precision, Recall, F1)
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Raporu PDF'e eklemek için DataFrame'e çevirebilirsin
    df_report = pd.DataFrame(report).transpose()
    print(df_report)

    # 3. Karmaşıklık Matrisi (Confusion Matrix)
    print("Confusion Matrix oluşturuluyor...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin Edilen (Predicted)')
    plt.ylabel('Gerçek (Actual)')
    plt.title(f'{model_adi} - Confusion Matrix')

    # Matrisleri kaydet
    os.makedirs("analiz_sonuclari", exist_ok=True)
    plt.savefig(f"analiz_sonuclari/{model_adi}_confusion_matrix.png")
    print(f"'{model_adi}_confusion_matrix.png' kaydedildi.")
    # plt.show() # Jupytersız çalışıyorsan bunu kullanma

print("\nTüm analizler tamamlandı.")