# Text Extraction with BERT

## Data - Twitter Sentiment Extraction

Each row contains the `text` of a tweet and a `sentiment` label. In the training set you are provided with a word or phrase drawn from the tweet (`selected_text`) that encapsulates the provided sentiment.

So, essentially we have the text and we have to extract the key parts which cause the sentiment.

### What do we extactly have to do?

We are given the text and the sentiment value and we are required to find the phrase from the text which contribute to the sentiment.

## How to approach??

### Creating the dataloader...

Let's say we have a text, and we have to extract phrases or words from the text which contribute to the sentiment.


- If the sentiment is `neutral` it's likely that the result is the original step itself. This is because we don't have definitive evidence that there would be a phrase of word which contribute the the `neutral` behaviour.

Text : $$w_1, w_2, w_3, ..., w_n$$

- Now, we create a vector where every entry represents the word in the original sentence. 
    - But, when we use tokenizers which use `Sub Word Tokenization`, the often  get split into smaller parts.

- We will essentially build up two vectors the `start` vector and the `end` vector.
    - These vectors will let us know where the context starts and where it ends.

For the dataloader, we need the
1. `input_ids`
2. `token_type_ids`
3. `attention_mask`
4. `start_index`
5. `end_index`
6. `sentiment`

### Model

- For building the model we have two approaches, 
    - one would be to use a one-hot vector for the start and end tokens and then use `nn.BCEWithLogitsLoss()` on the start output and the end outputs.
    - second would be to do something like a multiclass classification, this can be done if the start and end tokens are just indices.
        - In this case, it would be using `nn.CrossEntropyLoss()` for this scenario.


