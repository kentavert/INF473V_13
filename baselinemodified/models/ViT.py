import torchvision
import torch.nn as nn


class vitfinetune(nn.Module):
    def __init__(self, num_classes, frozen=False):
        super().__init__()
        self.backbone = torchvision.models.vit_b_16(pretrained=True)
        
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x