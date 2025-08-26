from torch import nn
import torch
from models.decoder import Decoder
from models.encoder import Encoder


class Transformer(nn.Module):
    def __init__(self,enc_vocab_size,dec_vocab_size,q_k_size,v_size,emb_size,head,f_size,nblocks):
        super().__init__()
        self.encoder=Encoder(enc_vocab_size,q_k_size,v_size,emb_size,head,f_size,nblocks)
        self.decoder=Decoder(dec_vocab_size,q_k_size,v_size,emb_size,head,f_size,nblocks)

    def forward(self,encoder_x,decoder_x):
        encoder_z=self.encode(encoder_x)
        return self.decode(decoder_x,encoder_z,encoder_x)

    def encode(self,encoder_x):
        encoder_z=self.encoder(encoder_x)
        return encoder_z

    def decode(self,decoder_x,encoder_z,encoder_x):
        decoder_z=self.decoder(decoder_x,encoder_z,encoder_x)
        return decoder_z