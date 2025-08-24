from torch import nn
import torch
from decoder import Decoder
from encoder import Encoder


class Transformer(nn.Module):
    def __init__(self,enc_vocab_size,dec_vocab_size,emb_size,q_k_size,v_size,f_size,head,nblocks,dropout=0.1,seq_max_len=5000):
        super().__init__()
        self.encoder=Encoder(enc_vocab_size,emb_size,q_k_size,v_size,f_size,head,nblocks,dropout,seq_max_len)
        self.decoder=Decoder(dec_vocab_size,emb_size,q_k_size,v_size,f_size,head,nblocks,dropout,seq_max_len)

    def forward(self,encoder_x,decoder_x):
        encoder_z=self.encode(encoder_x)
        return self.decode(decoder_x,encoder_z,encoder_x)

    def encode(self,encoder_x):
        encoder_z=self.encoder(encoder_x)
        return encoder_z

    def decode(self,decoder_x,encoder_z,encoder_x):
        decoder_z=self.decoder(decoder_x,encoder_z,encoder_x)
        return decoder_z