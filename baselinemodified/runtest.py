import cutout
import torch
cut = cutout.Cutout(1, 3)
imgs = torch.rand(2,3,5,5)
imgs = cut(imgs)
torch.optim.SGD(lr=1e-2,momentum=0.9,weight_decay=1,nesterov=True)
print(imgs.shape)
print(imgs)