import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from veri_hazirlama import get_dataloaders
from telos_model import teloCNN, get_parameter_count

# --- Ayarlar ---
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ADI = "teloCNN"

print(f"--- {MODEL_ADI} Özel Eğitimi Başlıyor ---")

# ÖNEMLİ: Burada standart 224 istiyoruz
train_loader, test_loader, NUM_CLASSES = get_dataloaders(batch_size=BATCH_SIZE, image_size=224)

model = teloCNN(num_classes=NUM_CLASSES)
print(f"Model Parametre Sayısı: {get_parameter_count(model):,}")
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
# Kendi modelinde tüm parametreler eğitilebilir, filter'a gerek yok ama kalsa da olur
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
writer = SummaryWriter(f'runs/fer2013_{MODEL_ADI}')

for epoch in range(EPOCHS):
    # Eğitim
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = 100 * correct / total

    # Test
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = 100 * correct_test / total_test

    print(
        f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.3f} Acc: {train_acc:.2f}% | Test Loss: {test_loss:.3f} Acc: {test_acc:.2f}%")

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', test_acc, epoch)

writer.close()
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), f"saved_models/{MODEL_ADI}.pth")
print("TeloCNN Eğitimi Tamamlandı ve Kaydedildi.")