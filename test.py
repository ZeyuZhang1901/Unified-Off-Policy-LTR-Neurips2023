import torch
import torch.nn.functional as F
from torch.nn.modules.transformer import MultiheadAttention

# Define the model parameters
sequence_length = 10
batch_size = 1
embedding_size = 8
num_heads = 2

# Create the input tensors
query = torch.randn(sequence_length, batch_size, embedding_size)
key = torch.randn(sequence_length, batch_size, embedding_size)
value = torch.randn(sequence_length, batch_size, embedding_size)

# Create the multi-head attention layer
multihead_attention = MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads)

# Generate the output sequence
output, _ = multihead_attention(query, key, value)

# Print the output
print(output)
