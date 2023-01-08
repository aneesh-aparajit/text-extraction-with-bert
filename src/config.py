import transformers
import torch as T

training_file = '../input/tweet-sentiment-extraction/train.csv'
test_file = '../input/tweet-sentiment-extraction/test.csv'
submission_file = '../input/tweet-sentiment-extraction/submission.csv'
max_len = 128
train_batch_size = 16
valid_batch_size = 8
base_model = "bert-base-uncased"

tokenizer = transformers.BertTokenizer.from_pretrained(
    base_model, 
    do_lower_case=True
)

device = T.device("cuda" if T.has_cuda else "mps" if T.has_mps else "cpu")