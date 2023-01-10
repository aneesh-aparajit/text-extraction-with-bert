import torch as T
from torch import nn as nn, optim as optim
import torch.nn.functional as F

from transformers import BertModel

import config
from typing import Optional, Dict, List

def loss_fn(start_pred: T.Tensor, end_pred: T.Tensor, start_orig: T.Tensor, end_orig: T.Tensor, attn_mask: T.Tensor) -> T.Tensor:
    start_loss = nn.CrossEntropyLoss()(start_pred, start_orig)
    end_loss = nn.CrossEntropyLoss()(end_pred, end_orig)
    return (start_loss + end_loss) / 2


class TweetModel(nn.Module):
    def __init__(self, max_pos: int) -> None:
        super(TweetModel, self).__init__()
        self.bert = BertModel.from_pretrained(config.base_model)
        self.start_drop = nn.Dropout(p=0.3)
        self.end_drop = nn.Dropout(p=0.3)
        self.start_linear = nn.Linear(768, max_pos)
        self.end_linear = nn.Linear(768, max_pos)

    def forward(self,
        input_ids: T.Tensor,
        token_type_ids: T.Tensor,
        attention_mask: T.Tensor,
        sentiment: T.Tensor,
        start: Optional[T.Tensor] = None,
        end: Optional[T.Tensor] = None) -> T.Tensor:
        x = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

        pred_start = self.start_drop(x)
        pred_end = self.end_drop(x)

        pred_start = self.start_linear(start)
        pred_end = self.end_linear(end)

        if targets is None:
            return start, end, None

        loss = loss_fn(pred_start, pred_end, start, end, attention_mask)

        return start, end, loss
