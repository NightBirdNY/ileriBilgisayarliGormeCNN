import torch
import torch.nn as nn
import torch.nn.functional as F

class teloCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(teloCNN, self).__init__()
        # Ödev yönergesine göre 3 kanallı (3, 48, 48) girdi alıyoruz

        # Katman 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # BatchNorm eklemek genellikle iyi fikirdir
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25) # Dropout eklemek ezberlemeyi (overfitting) azaltır

        # Katman 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        # Katman 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.25)

        # Boyutları düzleştirdikten (Flatten) sonraki tam bağlantı (FC) katmanları
        # 48x48 -> Pool1 -> 24x24 -> Pool2 -> 12x12 -> Pool3 -> 6x6
        # Düzleştirilmiş boyut: 128 * 6 * 6 = 4608
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc_dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Katman 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)

        # Katman 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)

        # Katman 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)

        # Düzleştirme
        x = x.view(-1, 128 * 28 * 28) # Boyutu (batch_size, 4608) yap

        # FC Katmanları
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = self.fc2(x)

        return x

# Parametre sayısını hesaplamak için bir yardımcı fonksiyon
def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Bu dosyayı doğrudan çalıştırırsan modeli test etmek için:
if __name__ == '__main__':
    model = teloCNN(num_classes=7)
    print(f"Model Mimarisi:")
    print(model)

    print(f"\nToplam Parametre Sayısı: {get_parameter_count(model):,}")

    # Test girdisi (Batch=1, Kanal=3, Yükseklik=48, Genişlik=48)
    test_tensor = torch.randn(1, 3, 48, 48)
    output = model(test_tensor)
    print(f"\nTest Çıktı Boyutu: {output.shape}")