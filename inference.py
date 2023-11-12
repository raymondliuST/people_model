from ml_dataset import mlDataset, mlDataModule
from bert_model.bert import BERT
import torch
import yaml
import numpy as np

def main(input_list):
    
    with open("config/pm.yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    dataset = mlDataset(config["data_params"])
    vocab_sizes = dataset.__getVocabSizes__()

    model = BERT(vocab_sizes)

    model_input_format = []
    for input in input_list:

        input_onehot = dataset.__stringToOnehot__(input)
        input_onehot_dict = {}
        for io in range(len(input_onehot)):
            input_onehot_dict[dataset.categorical_columns[io]] = torch.tensor(input_onehot[io]).clone().detach()

        label = None
        masked_position = torch.tensor([1 if e == -1 else 0 for e in input])


        d = {
            "input": input_onehot_dict,
            "label": label,
            "masked_position": masked_position
        }
        model_input_format.append(d)

    with torch.no_grad():
        output_logits = model(model_input_format)

        output_str = []
        for b in range(len(output_logits)):

            input_b = model_input_format[b]["input"]
            input_masks = model_input_format[b]["masked_position"]
            output_logits_b = output_logits[b]

            output_indices = []
            for i, col_name in enumerate(output_logits_b.keys()):
                if input_masks[i] == 0: # not masked -> append original token
                    output_index = torch.argmax(input_b[col_name]).item()
                else:
                    output_index = torch.argmax(output_logits_b[col_name]).item()
                output_indices.append(output_index)

            output_str.append(dataset.__indexToString__(output_indices))

    print(output_str)
    return output_str

    

if __name__ == "__main__":
    input_list = [[-1, 'Smartphone', 'iOS', 'Spain'],
                  ["SeaMonkey", 'Smartphone', -1, 'Spain'],
                  ['Mobile Firefox', 'Tablet', 'iOS 7', '-1']
                  ]
    
    main(input_list)