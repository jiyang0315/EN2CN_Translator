from torch import nn
import torch
from configs.config import MAX_SEQ_LEN
from data.dataset import EN2CNDataset
from transformers import BertTokenizer
import math


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        position_idx = torch.arange(0, MAX_SEQ_LEN, dtype=torch.float).unsqueeze(-1)
        position_emb_fill=position_idx*torch.exp(-torch.arange(0,embedding_dim,2)*math.log(10000.0)/embedding_dim)
        self.position_emb = torch.zeros(MAX_SEQ_LEN, embedding_dim)
        self.position_emb[:, 0::2] = torch.sin(position_emb_fill)
        self.position_emb[:, 1::2] = torch.cos(position_emb_fill)
        # self.register_buffer('position_emb',self.position_emb)  # 固定参数，训练时不改变
        
    def forward(self, x): # x->(batch_size, seq_len)
        emb = self.emb(x)  # x->(batch_size, seq_len, embedding_dim)
        return emb + self.position_emb.unsqueeze(0)[:,:x.size()[1],:]
        
        
if __name__=='__main__':
    
    en_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    en2cn = EN2CNDataset()
    en, cn = en2cn.__getitem__(0)
    en_id = en_tokenizer.encode(en)
    en_id_tensor = torch.tensor(en_id, dtype=torch.long).unsqueeze(0)
    
    # vocab_size=30522
    vocab_size = en_tokenizer.total_vocab_size
    embedding_dim = 512
    emb = Embedding(vocab_size, embedding_dim)
    print(emb(en_id_tensor))
    