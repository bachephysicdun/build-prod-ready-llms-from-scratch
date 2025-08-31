import torch
from transformer.model import Transformer

SOS_token = 0
EOS_token = 1
PAD_token = 2

index2words = {
    SOS_token: 'SOS',
    EOS_token: 'EOS',
    PAD_token: 'PAD'
}

words = "How are you doing ? I am good and you ?"
words_list = set(words.lower().split(' '))
for word in words_list:
    index2words[len(index2words)] = word
    
words2index = {w: i for i, w in index2words.items()}

def convert2tensors(sentence, max_len):
    words_list = sentence.lower().split(' ')
    padding = ['PAD'] * (max_len - len(words_list))
    words_list.extend(padding)
    indexes = [words2index[word] for word in words_list]
    return torch.tensor(indexes, dtype=torch.long).view(1, -1)


HIDDEN_SIZE = 12
VOCAB_SIZE = len(words2index)
N_BLOCKS = 10
D_FF = 20
CONTEXT_SIZE = 100
WINDOW_SIZE = 11
NUM_HEADS = 3
NUM_EXPERTS = 10
N_EXPERTS_PER_TOKEN = 2

transformer = Transformer(
    vocabulary_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE, 
    num_heads=NUM_HEADS, 
    window_size=WINDOW_SIZE, 
    d_ff=D_FF, 
    num_experts=NUM_EXPERTS, 
    n_experts_per_token=N_EXPERTS_PER_TOKEN, 
    n_blocks=N_BLOCKS,
    max_seq_len=CONTEXT_SIZE
)

input_sentence = "How are you doing ?"
output_sentence = "I am good and"

input_tensor = convert2tensors(input_sentence, CONTEXT_SIZE)

output = transformer(input_tensor)

_, indexes = output.squeeze().topk(1)
predicted_token = index2words[indexes[5].item()]

print(predicted_token)