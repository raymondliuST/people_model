import torch.nn as nn

from .transformer import TransformerBlock
from .embedding import BERTEmbedding
import torch

class classificationHead(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem
    """

    def __init__(self, hidden, vocab_sizes):
        """
        :param hidden: output size of the Transformer model
        :param vocab_sizes: List of vocabulary sizes for each attribute
        """
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.linear_layers = nn.ModuleDict({
            col_name: nn.Linear(hidden, size) for col_name, size in self.vocab_sizes.items()
        })
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        attribute_outputs = {}

        index = 0
        for col_name, linear_layer in self.linear_layers.items():
            attribute_outputs[col_name] = self.softmax(linear_layer(x[:,index]))
            index+=1

        list_of_ouputs = []

        for i in range(len(x)):
            single_output = {}
            for col_name in self.vocab_sizes.keys():
                single_output[col_name] = attribute_outputs[col_name][i:i+1]
            list_of_ouputs.append(single_output)

        return list_of_ouputs
    

class BERT(nn.Module):
    """
    People Model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_sizes, hidden=512, n_layers=1, attn_heads=8, dropout=0.1):
        """
        :param vocab_size: vocab_size of each column [list]
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.vocab_sizes = vocab_sizes

        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        # self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network

        # convert input array into uniform sizes
        self.linear_layers = nn.ModuleList([
            nn.Linear(size, 512) for col_name, size in vocab_sizes.items()
        ])

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        
        self.head = classificationHead(hidden, vocab_sizes)

    def forward(self, batch):
        # # attention masking for padded token
        # # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # # embedding the indexed sequence to sequence of vectors
        # x = self.embedding(x, segment_info)

        device = batch[0]["masked_position"].device

        stacked_batch_inputs = []
        batch_masked_position = []
        for x in batch:

            inputs = x["input"]
            masked_position = x["masked_position"] # i.e [0, 1, 0, 1]
            stacked_inputs = []
            i = 0
            for col_name, col_input in inputs.items():
                device = col_input.float().device

                if masked_position[i] == 0: 
                    stacked_inputs.append(self.linear_layers[i](col_input.float()))
                else: 
                    stacked_inputs.append(torch.full((512,), float(10^5), device = device))
                i+=1

            stacked_inputs = torch.stack(stacked_inputs, dim = 0) 
            stacked_batch_inputs.append(stacked_inputs)
            batch_masked_position.append(masked_position)

        stacked_batch_inputs = torch.stack(stacked_batch_inputs, dim = 0) # dim = [batch, # attributes, 512]
        batch_masked_position = torch.stack(batch_masked_position, dim = 0)
        
        # creating attention mask
        batch_size, sequence_length = batch_masked_position.shape
        attention_masks = []

        for i in range(batch_size):
            sequence_mask = batch_masked_position[i]
            attention_mask = torch.zeros(sequence_length, sequence_length, device=device)

            for j in range(sequence_length):
                for k in range(sequence_length):
                    if sequence_mask[j] == 0 and sequence_mask[k] == 0:
                        attention_mask[j][k] = 1
                    else:
                        attention_mask[j][k] = 0  # Set a very large negative value for masked positions

            attention_masks.append(attention_mask.unsqueeze(0))

        attention_masks = torch.stack(attention_masks, dim = 0)
        

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            stacked_batch_inputs = transformer.forward(stacked_batch_inputs, attention_masks)

        output = self.head(stacked_batch_inputs)

        return output
    
    def loss_function(self, batch, output):
        """
            output: list of dict; each dict is keyed by col names, valued by logits
        """

        losses = {}

        for col_name, size in self.vocab_sizes.items():

            target_matrix = torch.stack([b["label"][col_name].unsqueeze(0) for b in batch]).view(-1)
            prediction_matrix = torch.stack([b[col_name] for b in output]).view(-1, size)
            loss = nn.NLLLoss(ignore_index=-100)(prediction_matrix, target_matrix)

            losses[col_name] = loss

        return losses
    

