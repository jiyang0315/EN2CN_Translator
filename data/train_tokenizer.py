from tokenizer import BPETokenizer
from config import *

if __name__=='__main__':
    tokenizer = BPETokenizer()
    en_list = []
    cn_list = []
    with open('./train_data/train.en', 'r', encoding='utf-8') as fen, open('./train_data/train.zh', 'r', encoding='utf-8') as fcn:
        for en in fen:
            en_list.append(en.strip())
        for cn in fcn:
            cn_list.append(cn.strip())
    vocab_size = 10000
    tokenizer.add_special_tokens([IM_START,IM_END,BOS,EOS,PAD])
    tokenizer.train(en_list, vocab_size)
    tokenizer.save('./tokenizer.bin')
        