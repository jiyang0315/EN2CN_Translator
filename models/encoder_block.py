from torch import nn
from models.multi_attention import MultiHeadAttention
from data.dataset import EN2CNDataset
from transformers import BertTokenizer
from models.embeddings import Embedding
from configs.config import MAX_SEQ_LEN
import torch


class EncoderBlock(nn.Module):
    def __init__(self, vocab_size, q_k_size, v_size, embedding_dim, num_head, fn):
        super().__init__()
        self.attention = MultiHeadAttention(vocab_size, q_k_size, v_size, embedding_dim, num_head)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.feedward = nn.Sequential(nn.Linear(embedding_dim, fn), nn.ReLU(),nn.Linear(fn,embedding_dim), nn.Dropout())
        
    def forward(self, x, attn_mask): # x (batch_size, seq_len, embedding_dim)
        attention = self.attention(x,x, attn_mask)
        attention_out = self.layer_norm1(x+attention)
        feed_ward_out = self.layer_norm2(attention_out + self.feedward(attention_out))
        return feed_ward_out
        
    
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
    fn = 512
    emb = Embedding(vocab_size, embedding_dim)
    
    en_emb = emb(en_id_tensor)
    attn_mask = torch.zeros((1, en_emb.size()[1], en_emb.size()[1]))
    encoder_block = EncoderBlock(vocab_size, q_k_size, v_size, embedding_dim, num_head, fn)
    encoder_block(en_emb,attn_mask )    
    