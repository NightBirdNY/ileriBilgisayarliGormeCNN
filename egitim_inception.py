import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from torch.utils.tensorboard import SummaryWriter
from veri_hazirlama import get_dataloaders
from modeller import get_model

# --- Ayarlar ---
# Inception için batch size'ı biraz düşürmek gerekebilir (GPU belleği için)
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ADI = "inception_v3"

print(f"--- {MODEL_ADI} Özel Eğitimi Başlıyor ---")

# ÖNEMLİ: Burada 299 istiyoruz
train_loader, test_loader, NUM_CLASSES = get_dataloaders(batch_size=BATCH_SIZE, image_size=299)

model = get_model(MODEL_ADI, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
writer = SummaryWriter(f'runs/fer2013_{MODEL_ADI}_299')

# --- Eğitim Döngüsü (Inception Özel) ---
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        # Inception aux_logits çıktısı verir
        outputs, aux_outputs = model(inputs)
        loss1 = criterion(outputs, labels)
        loss2 = criterion(aux_outputs, labels)
        loss = loss1 + 0.4 * loss2

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
            outputs = model(inputs)  # Eval modunda tek çıktı verir
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
print("InceptionV3 Eğitimi Tamamlandı ve Kaydedildi.")