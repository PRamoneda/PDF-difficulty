import copy
import math
import os
import pdb

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence as packer, pad_packed_sequence as padder, PackedSequence
#from transformers.modeling_utils import SequenceSummary



import torch
import torch.nn.functional as F
import numpy as np

from pdf_difficulty.bootleg_gpt2 import GPTConfig, GPT, new_gelu


def prediction2label(pred):
    """Convert ordinal predictions to class labels, e.g.

    [0.9, 0.1, 0.1, 0.1] -> 0
    [0.9, 0.9, 0.1, 0.1] -> 1
    [0.9, 0.9, 0.9, 0.1] -> 2
    etc.
    """
    return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1

class ordinal_loss(nn.Module):
    """Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf"""
    def __init__(self, weight_class=False):
        super(ordinal_loss, self).__init__()
        self.weights = weight_class

    def forward(self, predictions, targets):
        # Fill in ordinalCoefficientVariationLoss target function, i.e. 0 -> [1,0,0,...]
        modified_target = torch.zeros_like(predictions)
        for i, target in enumerate(targets):
            modified_target[i, 0:target+1] = 1

        # if torch tensor is empty, return 0
        if predictions.shape[0] == 0:
            return 0
        # loss
        if self.weights is not None:
            # pdb.set_trace()
            return torch.sum((self.weights * F.mse_loss(predictions, modified_target, reduction="none")).mean(axis=1))
        else:
            return torch.sum((F.mse_loss(predictions, modified_target, reduction="none")).mean(axis=1))

class ContextAttention(nn.Module):
    def __init__(self, size, num_head):
        super(ContextAttention, self).__init__()
        self.attention_net = nn.Linear(size, size)
        self.num_head = num_head

        if size % num_head != 0:
            raise ValueError("size must be dividable by num_head", size, num_head)
        self.head_size = int(size / num_head)
        self.context_vector = torch.nn.Parameter(torch.Tensor(num_head, self.head_size, 1))
        nn.init.uniform_(self.context_vector, a=-1, b=1)

    def get_attention(self, x):
        attention = self.attention_net(x)
        attention_tanh = torch.tanh(attention)
        # attention_split = torch.cat(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
        attention_split = torch.stack(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
        similarity = torch.bmm(attention_split.view(self.num_head, -1, self.head_size), self.context_vector)
        similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1, 2, 0)
        return similarity

    def forward(self, x):
        attention = self.attention_net(x)
        attention_tanh = torch.tanh(attention)
        if self.head_size != 1:
            attention_split = torch.stack(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
            similarity = torch.bmm(attention_split.view(self.num_head, -1, self.head_size), self.context_vector)
            similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1, 2, 0)
            similarity[x.sum(-1) == 0] = -1e10  # mask out zero padded_ones
            softmax_weight = torch.softmax(similarity, dim=1)

            x_split = torch.stack(x.split(split_size=self.head_size, dim=2), dim=2)
            weighted_x = x_split * softmax_weight.unsqueeze(-1).repeat(1, 1, 1, x_split.shape[-1])
            attention = weighted_x.view(x_split.shape[0], x_split.shape[1], x.shape[-1])
        else:
            softmax_weight = torch.softmax(attention, dim=1)
            attention = softmax_weight * x

        sum_attention = torch.sum(attention, dim=1)
        return sum_attention


class gru_squared(nn.Module):
    def __init__(self, input_size, dropout, num_layers, hidden_size, batch_first, bidirectional):
        super(gru_squared, self).__init__()
        self.gru1 = nn.GRU(
            input_size=input_size,
            num_layers=num_layers,
            dropout=dropout,
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional
        )

    def forward(self, x, x_lengths):
        x_packed = packer(x.float(), x_lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        output, _ = self.gru1(x_packed)
        output_padded, _ = padder(output, batch_first=True)
        return output_padded


class dropout_gru_reduced_relu(nn.Module):
    def __init__(self, input_size=10, hidden_size=16, num_layers=3):
        super(dropout_gru_reduced_relu, self).__init__()
        self.dropout = nn.Dropout(p=0.2)

        self.gru1 = gru_squared(
            input_size=input_size,
            num_layers=num_layers,
            dropout=0.2,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x, x_lengths):
        x = self.dropout(x)
        output = self.gru1(x, x_lengths)
        return output

class PerformanceSummariser_ordinal(nn.Module):
    def __init__(self, n_classes=3, input_size=10):
        super(PerformanceSummariser_ordinal, self).__init__()
        #  input is (batchSize x time x 64)
        size = 64
        self.branch_rh = dropout_gru_reduced_relu(input_size=input_size, hidden_size=size//2, num_layers=3)
        self.attention_rh = ContextAttention(size=size, num_head=1)
        self.non_linearity = nn.ReLU()
        self.FC = nn.Linear(size, n_classes)

    def forward(self, fng_rh, rh_length):
        fng_rh = self.branch_rh(fng_rh, rh_length)
        summary_rh = self.attention_rh(fng_rh)
        classification = self.non_linearity(summary_rh)
        classification = self.FC(classification)
        return classification

class attention_classiffier(nn.Module):
    def __init__(self, n_classes=3, input_size=10):
        super(attention_classiffier, self).__init__()
        #  input is (batchSize x time x 64)
        self.attention_rh = ContextAttention(size=input_size, num_head=1)
        self.non_linearity = nn.ReLU()
        self.FC = nn.Linear(input_size, n_classes)

    def forward(self, fng_rh, rh_length):
        summary_rh = self.attention_rh(fng_rh)
        classification = self.non_linearity(summary_rh)
        classification = self.FC(classification)
        return classification


class PerformanceSummariser_ordinal_multi(nn.Module):
    def __init__(self):
        super(PerformanceSummariser_ordinal_multi, self).__init__()
        #  input is (batchSize x time x 64)
        size = 64
        self.branch_rh = dropout_gru_reduced_relu(input_size=size, hidden_size=size//2, num_layers=3)

        self.attention_rh = ContextAttention(size=size, num_head=1)
        self.non_linearity = nn.ReLU()
        self.FC_CIPI = nn.Linear(size, 9)
        self.FC_PS = nn.Linear(size, 9)
        self.FC_PL = nn.Linear(size, 9)
        self.FC_FS = nn.Linear(size, 5)

    def forward(self, bscore, length):
        bscore_encoded = self.branch_rh(bscore, length)
        summary_rh = self.attention_rh(bscore_encoded)
        summary_relued = self.non_linearity(summary_rh)
        c_CIPI = self.FC_CIPI(summary_relued)
        c_PS = self.FC_PS(summary_relued)
        c_PL = self.FC_PL(summary_relued)
        c_FS = self.FC_FS(summary_relued)
        return c_CIPI, c_PS, c_PL, c_FS


class identity_function(nn.Module):
    def __init__(self):
        super(identity_function, self).__init__()

    def forward(self, x):
        return x

class gpt2_classifier(nn.Module):
    def __init__(self, n_classes, checkpoint_dir="../bootlegGPT2/gpt2-imslp-ft_2fc", device=None):
        super(gpt2_classifier, self).__init__()
        self.n_classes = n_classes

        # resume training from a checkpoint.
        device = torch.device(device)
        ckpt_path = os.path.join(checkpoint_dir, 'ckpt.pt')
        print(ckpt_path, os.system("pwd"))
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        checkpoint_model_args["bias"] = False

        gptconf = GPTConfig(**checkpoint_model_args)
        model = GPT(gptconf)
        # load the model state
        self.transformer = model.transformer
        self.lm_head = nn.Linear(768, n_classes)
        self.cls_token = torch.randn(768, requires_grad=True)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, lengths):
        device = x.device
        b, t, embed_size = x.size()
        assert t <= 256, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)
        # forward the GPT model itself
        tok_emb = self.transformer.wte(x.float())  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd) # position embeddings of shape (1, t, n_embd)
        for idx in range(x.size()[0]):
            tok_emb[idx][lengths[idx] - 1] = self.cls_token

        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        mask = torch.zeros((b, t)).to(device)
        mask[torch.arange(b), lengths -1] = 1
        mask = mask.unsqueeze(-1)
        only_clas_tokens = torch.sum(mask * x, dim=1)
        only_clas_tokens = self.dropout(only_clas_tokens)
        return self.lm_head(only_clas_tokens)



class gpt2_classiffier_2_multi(gpt2_classifier):
    def __init__(self, checkpoint_dir="../bootlegGPT2/gpt2-imslp-ft_2fc", device=None):
        super(gpt2_classiffier_2_multi, self).__init__(1, checkpoint_dir, device)
        self.transformer_1 = copy.deepcopy(self.transformer.h[-1])
        self.transformer_ln_1 = copy.deepcopy(self.transformer.ln_f)
        del self.transformer
        self.dropout = nn.Dropout(p=0.2)
        del self.lm_head
        self.proj = nn.Linear(768, 768)
        self.FC_CIPI = nn.Linear(768, 9)
        self.FC_PS = nn.Linear(768, 9)
        self.FC_FS = nn.Linear(768, 5)

    def forward(self, x, lengths):
        device = x.device
        b, t, embed_size = x.size()
        for idx in range(x.size()[0]):
            x[idx][lengths[idx] - 1] = self.cls_token

        x = self.transformer_1(x)
        x = self.transformer_ln_1(x)

        mask = torch.zeros((b, t)).to(device)
        mask[torch.arange(b), lengths -1] = 1
        mask = mask.unsqueeze(-1)
        only_clas_tokens = torch.sum(mask * x, dim=1)
        projected = self.proj(only_clas_tokens)
        projected = new_gelu(projected)
        return self.FC_CIPI(projected), self.FC_PS(projected), self.FC_FS(projected)

    def get_projection(self, x, lengths):

        device = x.device
        b, t, embed_size = x.size()
        self.cls_token.requires_grad = False
        for idx in range(x.size()[0]):
            x[idx][lengths[idx] -1] = self.cls_token

        x = self.transformer_1(x)
        x = self.transformer_ln_1(x)

        mask = torch.zeros((b, t)).to(device)
        mask[torch.arange(b), lengths - 1] = 1
        mask = mask.unsqueeze(-1)
        only_clas_tokens = torch.sum(mask * x, dim=1)
        projected = self.proj(only_clas_tokens)
        # projected = new_gelu(projected)
        return projected

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe


