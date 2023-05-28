import cutout
import torch
cut = cutout.Cutout(1, 3)
imgs = torch.rand(2,3,5,5)
imgs = cut(imgs)
print(imgs.shape)
print(imgs)