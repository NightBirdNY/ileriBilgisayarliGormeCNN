import veri_hazirlama
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
import kagglehub
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import tensorboard
from veri_hazirlama import get_dataloaders
from modeller import get_model

# --- Parametreler ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Kullanılacak cihaz: {DEVICE}")
train_loader, test_loader, NUM_CLASSES = get_dataloaders(batch_size=BATCH_SIZE)
model_listesi = ["resnet50", "alexnet", "vgg16", "vgg19", "densenet121", "efficientnet_b0"]

for model_adi in model_listesi:
    print(f"\n--- Model Eğitiliyor: {model_adi} ---")

    # modeli çağırın
    model = get_model(model_adi, num_classes=7)

    # Modeli GPU'ya gönderin
    model = model.to(DEVICE)

    # O modele ait optimizatörü tanımlayın
    # SADECE eğitilebilir parametreleri optimize etmesini söylüyoruz
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    #Doldurulacak
    print(f"--- {model_adi} eğitimi bitti ---")