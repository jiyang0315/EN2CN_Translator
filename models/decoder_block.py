from torch import nn
from models.multi_attention import MultiHeadAttention
from data.dataset import EN2CNDataset
from transformers import BertTokenizer
from models.embeddings import Embedding
from configs.config import MAX_SEQ_LEN
import torch
from models.encoder_block import EncoderBlock


class DecoderBlock(nn.Module):
    def __init__(self, vocab_size, q_k_size, v_size, embedding_dim, num_head, fn):
        super().__init__()
        self.attention_1 = MultiHeadAttention(vocab_size, q_k_size, v_size, embedding_dim, num_head)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.layer_norm3 = nn.LayerNorm(embedding_dim)
        self.feedward = nn.Sequential(nn.Linear(embedding_dim, fn), nn.ReLU(),nn.Linear(fn,embedding_dim), nn.Dropout())
        self.attention_2 = MultiHeadAttention(vocab_size, q_k_size, v_size, embedding_dim, num_head)
    
    def forward(self,x, encode_z, first_attn_mask, second_attn_mask): # x (batch, seq_len, embedding_dim)  pad_mask= 
         attention_1 = self.attention_1(x, x,first_attn_mask)
         attention_1 = self.layer_norm1(x + attention_1)
         
         attention_2 = self.attention_2(attention_1, encode_z, second_attn_mask)
         attention_2 = self.layer_norm2(attention_2 + attention_1)
         
         feedward_out = self.layer_norm3(attention_2 + self.feedward(attention_2))
         
         return feedward_out
         
      