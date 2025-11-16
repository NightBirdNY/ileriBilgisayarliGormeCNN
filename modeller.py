
import timm
import torch.nn as nn

#models
import torchvision.models as models


def get_model(model_name: str, num_classes: int = 7):

    #İstenen SOTA modelini yükler, dondurur ve kafasını 7 sınıfa göre değiştirir.
    model = None

    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        giris_boyutu = model.fc.in_features
        model.fc = nn.Linear(giris_boyutu, num_classes)

    elif model_name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        giris_boyutu = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(giris_boyutu, num_classes)

    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        giris_boyutu = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(giris_boyutu, num_classes)

    elif model_name == "vgg19":
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        giris_boyutu = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(giris_boyutu, num_classes)

    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        giris_boyutu = model.classifier.in_features
        model.classifier = nn.Linear(giris_boyutu, num_classes)

    elif model_name == "efficientnet_b0":
        # Sizin düzelttiğiniz gibi 'create_model' kullandım
        model = timm.create_model('efficientnet_b0', pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        giris_boyutu = model.classifier.in_features
        model.classifier = nn.Linear(giris_boyutu, num_classes)

    elif model_name == "inception_v3":
        # Sizin düzelttiğiniz gibi 'aux_logits=True' ve '.fc' kullandım
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT,
            aux_logits=True)  # Eğitim döngüsünde özel işlem gerektirecek
        for param in model.parameters():
            param.requires_grad = False
        giris_boyutu = model.fc.in_features
        model.fc = nn.Linear(giris_boyutu, num_classes)
        giris_boyutu_yardimci = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(giris_boyutu_yardimci, num_classes)

    else:
        raise ValueError(f"Bilinmeyen model adı: {model_name}")

    return model


