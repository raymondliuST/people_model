from dataset import mlDataset
import yaml
with open("config/pm.yaml", 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

dataset = mlDataset(config["data_params"])

import pdb
pdb.set_trace()