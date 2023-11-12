from bert_model.bert import BERT

from ml_dataset import mlDataset, mlDataModule
from pm_experiment import pmExperiment
import torch
import yaml
with open("config/pm.yaml", 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

torch.multiprocessing.set_sharing_strategy('file_system')

dataloader = mlDataModule(dataset_config=config["data_params"])
# vocab_sizes = dataloader.train_dataset.__getVocabSizes__()

# experiment = pmExperiment(config, vocab_sizes)

# for batch in dataloader.train_dataloader():
#     experiment.validation_step(batch, 0)
