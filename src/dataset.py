import torch as T
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import config

from typing import Dict, List
from pprint import pprint

class TweetDataset(Dataset):
    def __init__(self, text: List[str], selected_text: List[str], sentiment: List[str]) -> None:
        super(TweetDataset, self).__init__()
        self.text = text
        self.selected_text = selected_text
        self.sentiment = sentiment
        self.max_len = config.max_len
    
    def __len__(self) -> int:
        return len(self.text)
    
    def __getitem__(self, ix) -> Dict[str, T.Tensor]:
        text = self.text[ix].split()
        context = self.selected_text[ix].split()
        sentiment = self.sentiment[ix]

        '''
        Now we need to split the string into words and keep track of the words which belong to the selected text. 
            - One thing to notice here is that, here we include even the part is not in the continuous subarray, 
                - we go by the assumption that both may have the similar intent in the sentiment.
        - Firstly, start out by finding the start and end index of the selected text in the actual text.
            - it's better if we use the word based.
        '''
        start_ix, end_ix = 0, len(text) - 1
        len_context = len(context)
        for ix in range(len(text)):
            if text[ix] == context[0]:
                if text[ix:ix+len_context] == context:
                    start_ix = ix
                    end_ix = ix + len_context - 1

        print(f'start: {start_ix}')
        print(f'end: {end_ix}')
        
        input_ids = []
        attention_mask = []
        token_type_ids = []
        
        for ix,  word in enumerate(text):
            word_tok = config.tokenizer(word, add_special_tokens=False)
            len_tok = len(word_tok['input_ids'])
            input_ids.extend(word_tok['input_ids'])
            attention_mask.extend(word_tok['attention_mask'])
            token_type_ids.extend(word_tok['token_type_ids'])
            if ix < start_ix:
                start_ix += (len_tok - 1)
                end_ix += (len_tok - 1)
            elif ix < end_ix:
                end_ix += (len_tok - 1)
            else:
                continue
            print(f'start: {start_ix} end_ix: {end_ix}')
        
        enc_sentiment = [1, 0, 0] if sentiment == 'negative' else [0, 1, 0] if sentiment == 'neutral' else [0, 0, 1]

        input_ids = [101] + input_ids + [102]
        token_type_ids = [0] + token_type_ids + [0]
        attention_mask = [1] + attention_mask + [1]
        
        padding_len = config.max_len - len(input_ids)
        token_type_ids.extend([0] * padding_len)
        attention_mask.extend([0] * padding_len)
        input_ids.extend([0] * padding_len)

        return {
            'input_ids': T.tensor(input_ids, dtype=T.long),
            'token_type_ids': T.tensor(token_type_ids, dtype=T.long),
            'attention_mask': T.tensor(attention_mask, dtype=T.long), 
            'start': T.tensor(start_ix, dtype=T.long),
            'end': T.tensor(end_ix, dtype=T.long), 
            'sentiment': T.tensor(enc_sentiment, dtype=T.long), 
            'orig_sentiment': sentiment, 
            'text':  ' '.join(text),
            'context': ' '.join(context),
            'padding_len': padding_len
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
