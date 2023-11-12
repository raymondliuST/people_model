import lightning.pytorch as pl
import torch
import os
from lightning.pytorch.demos import Transformer
from ml_dataset import * 
from torch.utils.data import random_split, DataLoader

class mlDataModule(pl.LightningDataModule):
    def __init__(self, dataset_config):
        super().__init__()

        self.train_batch_size = dataset_config["train_batch_size"]
        self.val_batch_size = dataset_config["val_batch_size"]

        self.train_dataset = mlDataset(dataset_config, partition="train")
        self.val_dataset = mlDataset(dataset_config, partition="validation")

        self.num_workers = min(dataset_config["num_workers"], os.cpu_count())

        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)
    
    def collate_fn(self, data):
        return list(data)
