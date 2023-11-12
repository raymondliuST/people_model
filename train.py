import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pm_experiment import pmExperiment
from ml_dataset import mlDataModule

import yaml
import torch
from pytorch_lightning.loggers import WandbLogger  # Import WandbLogger

def main(dataset_config_path):
    # Define your model, dataset, and dataloader
    with open(dataset_config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    torch.multiprocessing.set_sharing_strategy("file_system") 


    dataloader = mlDataModule(config["data_params"])
    vocab_sizes = dataloader.train_dataset.__getVocabSizes__()

    model_module = pmExperiment(config, vocab_sizes)

    # Initialize WandbLogger with your project name and any other desired settings
    wandb_logger = WandbLogger(project="pm", log_model=True)

    # Define a Lightning Trainer with WandbLogger for experiment tracking
    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1,
        logger=wandb_logger,  # Use the WandbLogger for experiment tracking
        max_epochs=10,  # Number of training epochs
        progress_bar_refresh_rate=10,  # Update the progress bar every 10 batches
        log_every_n_steps=1,  # Log metrics every batch
        default_root_dir = "./checkpoints",
        gradient_clip_val=0.5
    )

    # Train the model
    trainer.fit(model_module, dataloader.train_dataloader(), dataloader.val_dataloader())

if __name__ == "__main__":
    main("config/pm.yaml")