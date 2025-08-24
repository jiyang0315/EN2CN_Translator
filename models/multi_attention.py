import torch
from torch import nn
from models.embeddings import Embedding
from configs.config import MAX_SEQ_LEN
from data.dataset import EN2CNDataset
from transformers import BertTokenizer
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, vocab_size, q_k_size, v_size, embedding_dim, num_head):
        super().__init__()
        self.w_q = nn.Linear(embedding_dim, q_k_size*num_head)
        self.w_k = nn.Linear(embedding_dim, q_k_size*num_head)
        self.w_v = nn.Linear(embedding_dim, v_size*num_head)
        self.num_head = num_head
        self.q_k_size = q_k_size
        self.v_size = v_size
        self.linear = nn.Linear(v_size*num_head, embedding_dim)
        
    def forward(self, x_q, x_k_v, attn_mask): # x [batch_size, seq_len, embedding_dim]    attn_mask: [batch_size,seq_len,seq_len]
        
        # q [batch_size, seq_len, q_k_size*num_head] ->[batch_size,num_head, seq_len, q_k_size]
        q = self.w_q(x_q).unsqueeze(2).view(x_q.size()[0], x_q.size()[1], self.num_head, self.q_k_size).permute(0,2,1,3)  
        
        # k [batch_size, seq_len, q_k_size*num_head] ->[batch_size, num_head, q_k_size, seq_len]
        k_T = self.w_k(x_k_v).unsqueeze(2).view(x_k_v.size()[0], x_k_v.size()[1], self.num_head, self.q_k_size).permute(0,2,3,1)  
        # atten_mask = 
        # attention [batch_size, num_head, seq_len, seq_len]
        attention = torch.matmul(q, k_T)/math.sqrt(self.q_k_size)
        attn_mask = attn_mask.unsqueeze(1).expand(-1,self.num_head,-1,-1)
        attention.masked_fill(attn_mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        
        v = self.w_v(x_k_v).unsqueeze(2).view(x_k_v.size()[0], x_k_v.size()[1], self.num_head, self.v_size).permute(0,2,1,3)
        
        attention = torch.matmul(attention, v).transpose(1,2).reshape(x_k_v.size(0),x_k_v.size(1), self.num_head*self.v_size)
        
        # 将输出的size 调整为与输入相同
        attention = self.linear(attention)
        return attention
        
        
        
if __name__=='__main__':
    
    en_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    en2cn = EN2CNDataset()
    en, cn = en2cn.__getitem__(0)
    en_id = en_tokenizer.encode(en)
    en_id_tensor = torch.tensor(en_id, dtype=torch.long).unsqueeze(0)
    
    # vocab_size=30522
    vocab_size = en_tokenizer.total_vocab_size
    embedding_dim = 512
    q_k_size = 256
    v_size = 512
    num_head = 8
    emb = Embedding(vocab_size, embedding_dim)
    
    en_emb = emb(en_id_tensor)
    attention = MultiHeadAttention(vocab_size, q_k_size, v_size,embedding_dim, num_head)
    
    attn_mask = torch.zeros((1, en_emb.size()[1], en_emb.size()[1]))
    en_attention = attention(en_emb,en_emb,attn_mask)
    print('en_attention',en_attention)