import torch
from torch import nn
from torchvision import models


def build_model(name, pretrained=True):
    builders = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    }
    builder, weights = builders[name]
    model = builder(weights=weights if pretrained else None)
    # Initialize a new binary classifier, as required by the lab.
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def choose_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
