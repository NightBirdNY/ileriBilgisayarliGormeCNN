from veri_hazirlama import get_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim
import kagglehub
import os
import time  # Zaman ölçümü için
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# TensorBoard için
from torch.utils.tensorboard import SummaryWriter
from veri_hazirlama import get_dataloaders
from modeller import get_model
from telos_model import teloCNN,get_parameter_count


# --- Parametreler ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 5  # Ödev için 10-20 epoch iyi bir başlangıç olabilir
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Kullanılacak cihaz: {DEVICE}")

# --- Veri Yükleyiciler ---
try:
    train_loader, test_loader, NUM_CLASSES = get_dataloaders(batch_size=BATCH_SIZE)
except Exception as e:
    print(f"Veri yüklenirken hata oluştu: {e}")
    print("Lütfen 'kaggle_check.py' ile API anahtarınızı kontrol edin.")
    exit()

# --- Model Listesi ---
# Görev 3'ü yapınca kendi modelini de eklersin, örn: "kendi_cnn"
model_listesi = ["resnet50", "alexnet", "vgg16", "vgg19", "densenet121", "efficientnet_b0", "inception_v3", "teloCNN"]

# --- Kayıp Fonksiyonu ---
criterion = nn.CrossEntropyLoss()


# --- Eğitim ve Test Fonksiyonları ---

def train_one_epoch(model, loader, optimizer, criterion, model_name):
    model.train()  # Modeli eğitim moduna al
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # optimizer gradyanları sıfırla
        optimizer.zero_grad()

        # --- İLERİ BESLEME ---
        # InceptionV3 özel durumu (aux_logits)
        if model_name == "inception_v3" and model.training:
            outputs, aux_outputs = model(inputs)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4 * loss2  # Önerilen birleşim
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # --- GERİ YAYILIM ---
        loss.backward()
        optimizer.step()

        # İstatistikler
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate_model(model, loader, criterion):
    model.eval()  # Modeli değerlendirme moduna al
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Gradyan hesaplamasını kapat
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # İstatistikler
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


# --- ANA EĞİTİM DÖNGÜSÜ ---

for model_adi in model_listesi:
    print(f"\n--- Model Eğitiliyor: {model_adi} ---")

    # TensorBoard writer'ı başlat (her model için ayrı bir klasör)
    writer = SummaryWriter(f'runs/fer2013_{model_adi}')

    # Modeli çağırın

    try:
        if model_adi == "teloCNN":
            model = teloCNN(num_classes=NUM_CLASSES)
            print(f"TeloCNN modeli Yüklendi. Parametre sayısı : {get_parameter_count(model):,}")
        else:
            model = get_model(model_adi, num_classes=NUM_CLASSES)
    except ValueError as e:
        print(e)
        continue

    # Modeli GPU'ya gönderin
    model = model.to(DEVICE)

    # Optimizatör
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    start_time = time.time()

    # Epoch döngüsü
    for epoch in range(EPOCHS):
        # Eğitim
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, model_adi)

        # Değerlendirme (Test)
        test_loss, test_acc = validate_model(model, test_loader, criterion)

        print(f"Epoch {epoch + 1}/{EPOCHS}.. "
              f"Train loss: {train_loss:.3f}.. "
              f"Train acc: {train_acc:.2f}%.. "
              f"Test loss: {test_loss:.3f}.. "
              f"Test acc: {test_acc:.2f}%")

        # TensorBoard'a logla
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)

    end_time = time.time()
    print(f"Toplam Eğitim Süresi ({model_adi}): {(end_time - start_time):.2f} saniye")
    writer.close()

    # Raporlama için modeli kaydedebilirsin
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), f"saved_models/{model_adi}.pth")

    print(f"--- {model_adi} eğitimi bitti ---")

print("\nTüm modellerin eğitimi tamamlandı.")