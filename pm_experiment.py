import os
import math
import torch
from torch import optim
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from torchmetrics.classification import MulticlassAccuracy
import torch.nn as nn

import numpy as np
from bert_model.bert import BERT


class pmExperiment(pl.LightningModule):

    def __init__(self,
                 config: dict, vocab_sizes) -> None:
        super(pmExperiment, self).__init__()

        self.model = BERT(vocab_sizes)
        self.params = config["exp_params"]
        self.vocab_sizes = vocab_sizes
        # self.curr_device = None

        # For batch loss calculation
        self.batch_train_loss = []

        # Accuracy matrixes: multi class
        self.micro_acc_val = nn.ModuleDict()
        for col_name, size in vocab_sizes.items():
            self.micro_acc_val[col_name] = MulticlassAccuracy(num_classes = size, average = 'micro', ignore_index = -100)

        self.macro_acc_val = nn.ModuleDict()
        for col_name, size in vocab_sizes.items():
            self.macro_acc_val[col_name] = MulticlassAccuracy(num_classes = size, average = 'macro', ignore_index = -100)

        # Accuracy matrixes: multi class train
        self.micro_acc_train = nn.ModuleDict()
        for col_name, size in vocab_sizes.items():
            self.micro_acc_train[col_name] = MulticlassAccuracy(num_classes = size, average = 'micro', ignore_index = -100)

        self.macro_acc_train = nn.ModuleDict()
        for col_name, size in vocab_sizes.items():
            self.macro_acc_train[col_name] = MulticlassAccuracy(num_classes = size, average = 'macro', ignore_index = -100)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)

    def training_step(self, batch):
        outputs = self.forward(batch)
        train_losses = self.model.loss_function(batch, outputs)

        total_losses = torch.sum(torch.stack([loss for col_name, loss in train_losses.items() if loss is not None]))


        self.log("train_loss_step", total_losses)
        self.log_dict({f"{key}_loss_step": val.item() for key, val in train_losses.items()}, sync_dist=True)
        self.log_dict({"total_train_loss_step": total_losses})

        self.batch_train_loss.append(total_losses.cpu().detach().numpy())

        # update accuracy
        
        
        for col_name, size in self.vocab_sizes.items():

            target_matrix = torch.stack([b["label"][col_name].unsqueeze(0) for b in batch]).view(-1)
            prediction_matrix = torch.stack([b[col_name] for b in outputs]).view(-1, size)
            
            micro_acc = self.micro_acc_train[col_name](prediction_matrix, target_matrix)
            macro_acc = self.macro_acc_train[col_name](prediction_matrix, target_matrix)

            self.log_dict({f"{col_name}_train_micro_acc_step": micro_acc}, on_step=True, on_epoch=False)
            self.log_dict({f"{col_name}_train_macro_acc_step": macro_acc}, on_step=True, on_epoch=False)

        return total_losses
    
    def on_train_epoch_end(self):
        # F1 Macro all epoch saving outputs and target per batch
        train_loss_epoch = np.mean(self.batch_train_loss)

        self.log("training_loss_epoch", train_loss_epoch, on_step=False, on_epoch=True, prog_bar=True)

        # free up the memory
        # --> HERE STEP 3 <--
        self.batch_train_loss.clear()
        
        for col_name, size in self.vocab_sizes.items():
            self.log_dict({f"{col_name}_train_micro_acc_epoch": self.micro_acc_train[col_name].compute()}, on_step=False, on_epoch=True)
            self.micro_acc_train[col_name].reset()

            self.log_dict({f"{col_name}_train_macro_acc_epoch": self.macro_acc_train[col_name].compute()}, on_step=False, on_epoch=True)
            self.macro_acc_train[col_name].reset()


    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        outputs = self.forward(batch)

        val_losses = self.model.loss_function(batch, outputs)

        total_losses = torch.sum(torch.stack([loss for col_name, loss in val_losses.items() if loss is not None]))

        self.log("val_loss_step", total_losses)
        self.log_dict({f"{key}_loss_step": val.item() for key, val in val_losses.items()}, sync_dist=True, on_step=True, on_epoch=False)
        self.log_dict({"total_val_loss_step": total_losses}, on_step=True, on_epoch=False)

        self.batch_train_loss.append(total_losses.cpu().detach().numpy())

        # update accuracy
        
        
        for col_name, size in self.vocab_sizes.items():
            
            target_matrix = torch.stack([b["label"][col_name].unsqueeze(0) for b in batch]).view(-1)
            prediction_matrix = torch.stack([b[col_name] for b in outputs]).view(-1, size)
            
            micro_acc = self.micro_acc_val[col_name](prediction_matrix, target_matrix)
            macro_acc = self.macro_acc_val[col_name](prediction_matrix, target_matrix)

            self.log_dict({f"{col_name}_val_micro_acc_step": micro_acc}, on_step=True, on_epoch=False)
            self.log_dict({f"{col_name}_val_macro_acc_step": macro_acc}, on_step=True, on_epoch=False)
          
        return total_losses

    def on_validation_epoch_end(self):
        for col_name, size in self.vocab_sizes.items():
            self.log_dict({f"{col_name}_val_micro_acc_epoch": self.micro_acc_val[col_name].compute()}, on_step=False, on_epoch=True)
            self.micro_acc_val[col_name].reset()

            self.log_dict({f"{col_name}_val_macro_acc_epoch": self.macro_acc_val[col_name].compute()}, on_step=False, on_epoch=True)
            self.macro_acc_val[col_name].reset()

        
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.02)
    
    

       