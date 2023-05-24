import torchvision
import torch.nn as nn


class vitfinetune(nn.Module):
    def __init__(self, num_classes, frozen=False):
        super().__init__()
        self.backbone = torchvision.models.maxvit_t(pretrained=True)
        self.backbone.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                                nn.Flatten(),
                                                nn.LayerNorm(512),)
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x