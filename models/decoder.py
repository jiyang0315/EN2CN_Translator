from torch import nn
import torch
from models.encoder_block import EncoderBlock
from embeddings import Embedding
from decoder_block import DecoderBlock

class Decoder(nn.Module):
    def __init__(self):
        super().__init__(self, vocab_size, q_k_size, v_size, embedding_dim, num_head, fn, num_decoder_block)
        self.emb = Embedding(vocab_size, embedding_dim)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(vocab_size, q_k_size, v_size, embedding_dim, num_head, fn)
            for _ in range(num_decoder_block)
        ])
        
        # 输出向量词概率Logits
        self.linear=nn.Linear(emb_size,vocab_size)  
        
    def forward(self, x, encoder_z, encoder_x): # x(batch_size, seq_len)
        
        first_attn_mask = (x == PAD_IDX).unsqueeze(1).expand(x.size(0), x.size(1), x.size(1) )   # (batch_size, seq_len, seq_len)
        first_attn_mask=first_attn_mask|torch.triu(torch.ones(x.size()[1],x.size()[1]),diagonal=1).bool().unsqueeze(0).expand(x.size()[0],-1,-1).to(DEVICE) # &目标序列的向后看掩码
        second_attn_mask=(encoder_x==PAD_IDX).unsqueeze(1).expand(encoder_x.size()[0],x.size()[1],encoder_x.size()[1]).to(DEVICE) # (batch_size,target_len,src_len)
        
        x=self.emb(x)
        for block in self.decoder_blocks:
            x=block(x,encoder_z,first_attn_mask,second_attn_mask)
        
        return self.linear(x) # (batch_size,target_len,vocab_size)