from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from hydra.utils import instantiate
import torch

import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torch

class unlabelledDataset(Dataset):
    def __init__(self, test_dataset_path, test_transform):
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        images_list = os.listdir(self.test_dataset_path)
        # filter out non-image files
        self.images_list = [image for image in images_list if image.endswith(".jpg")]
        # set a non exist label
        self.labels = [48] * len(self.images_list)#attention: 48 or 47
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        label = self.labels[idx]
        image_path = os.path.join(self.test_dataset_path, image_name)
        image = Image.open(image_path)
        image = self.test_transform(image)
        return image, label, idx

    def __len__(self):
        return len(self.images_list)
    
    def set_label(self, newlabel, idx):
         self.labels[idx] = newlabel
    
class DataModule:
    def __init__(
        self,
        train_dataset_path,
        unlabelled_dataset_path,
        train_transform,
        val_transform,
        aug_transform,
        batch_size,
        num_workers,
    ):
        self.dataset = ImageFolder(train_dataset_path, transform=train_transform)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset,
            [
                int(0.8 * len(self.dataset)),
                len(self.dataset) - int(0.8 * len(self.dataset)),
            ],
            generator=torch.Generator().manual_seed(3407),
        )
        self.aug_transform = aug_transform
        self.val_dataset.transform = val_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.unlabelled_dataset_path = unlabelled_dataset_path
        self.unlabelled_transform = train_transform
        self.unlabelled_dataset = unlabelledDataset(self.unlabelled_dataset_path, self.unlabelled_transform)

    def data_augment(self, pic):
            pic_augment = self.aug_transform(pic)
            return pic_augment

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    def unlabelled_dataloader(self):
         return DataLoader(self.unlabelled_dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=self.num_workers,
                        )