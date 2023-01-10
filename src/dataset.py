import torch as T
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import config

from typing import Dict, List
from pprint import pprint


class TweetDataset(Dataset):
    def __init__(self, texts: List[str], contexts: List[str], sentiments: List[str]) -> None:
        super(TweetDataset, self).__init__()
        self.texts = texts
        self.contexts = contexts
        self.sentiments = sentiments
        self.max_len = config.max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, ix: int) -> Dict[str, T.Tensor]:
        text = self.texts[ix]
        context = self.contexts[ix]
        sentiment = self.sentiments[ix]

        sentiment_tensor = [0, 1, 0]
        if sentiment == 'negative':
            sentiment_tensor = [1, 0, 0]
        elif sentiment == 'positive':
            sentiment_tensor = [0, 0, 1]

        text_tok = config.tokenizer.encode(text, add_special_tokens=False)
        context_tok = config.tokenizer.encode(context, add_special_tokens=False)

        start_ix, end_ix = -1, -1

        for i in range(len(text_tok)):
            if text_tok[i] == context_tok[i]:
                if text_tok[i:i+len(context_tok)] == context_tok:
                    start_ix = i
                    end_ix = i + len(context_tok) - 1
                    break

        input_ids = [101] + text_tok + [102]
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        # padding
        padding_len = self.max_len - len(input_ids)
        input_ids.extend([0] * padding_len)
        attention_mask.extend([0] * padding_len)
        token_type_ids.extend([0] * padding_len)

        return {
            'text_tok': T.tensor(text_tok, dtype=T.long),
            'context_tok': T.tensor(context_tok, dtype=T.long),
            'sentiment': T.tensor(sentiment_tensor, dtype=T.long),
            'start': start_ix,
            'end': end_ix,
            'input_ids': T.tensor(input_ids, dtype=T.long),
            'attention_mask': T.tensor(attention_mask, dtype=T.long),
            'token_type_ids': T.tensor(token_type_ids, dtype=T.long)
        }


if __name__ == '__main__':
    dfx = pd.read_csv(config.training_file)
    print(dfx.head())
    
    text = dfx['text'].values
    selected_text = dfx['selected_text'].values
    sentiments = dfx['sentiment'].values
    print("\n\n")

    dataset = TweetDataset(text=text, selected_text=selected_text, sentiment=sentiments)
    pprint(dataset[2])
