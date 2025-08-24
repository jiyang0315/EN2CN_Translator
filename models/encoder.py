from torch import nn
import torch
from models.encoder_block import EncoderBlock
from embeddings import Embedding
from encoder_block import EncoderBlock


class Encoder(nn.Module):
    def __init__(self, vocab_size, q_k_size, v_size, embedding_dim, num_head, fn, num_encoder_block):
        super().__init__()
        self.emb = Embedding(vocab_size, embedding_dim)
        self.num_encoder_block = num_encoder_block
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(vocab_size, q_k_size, v_size, embedding_dim, num_head, fn)
            for _ in range(num_encoder_block)
            ]
        )
        
    def forward(self, x):
        pad_mask = (x == PAD_IDX).unsqueeze(1).expand(x.size(0), x.size(1), x.size(1) )  # (batch, seq_len, seq_len)
        output = self.emb(x)  # (batch, seq_len) -> (batch, seq_len, embedding_dim)
        
        for block in self.encoder_blocks:
            output = block(output, pad_mask)
        
        return output