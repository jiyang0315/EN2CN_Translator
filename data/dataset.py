from torch.utils.data import Dataset
from transformers import BertTokenizer
from configs.config import *


class EN2CNDataset(Dataset):
    def __init__(self, en_tokenizer, cn_tokenizer, split='train'):
        super().__init__()
        self.en_list = []
        self.cn_list = []
        self.en_tokenizer = en_tokenizer
        self.cn_tokenizer = cn_tokenizer
        if split = 'train':
            en_data_path = 'data/train_data/train.en'
            cn_data_path = 'data/train_data/train.zh'
        else:
            en_data_path = 'data/val_data/valid.en-zh.en.sgm'
            cn_data_path = 'data/val_data/valid.en-zh.zh.sgm'
            
        with open(en_data_path, 'r', encoding='utf-8') as fen, open(cn_data_path, 'r', encoding='utf-8') as fcn:
            for en in fen:
                self.en_list.append(BOS  + en.strip() + EOS)
            for cn in fcn:
                self.cn_list.append(BOS + cn.strip() + EOS)
        assert len(self.en_list)==len(self.cn_list)
        
    def __len__(self):
        return len(self.en_list)
    
    def __getitem__(self, index):
        # add_special_tokens=False  防止tokenizer 自动加入开始和结尾
        return self.en_tokenizer.encode(self.en_list[index], add_special_tokens=False), self.cn_tokenizer.encode(self.cn_list[index], add_special_tokens=False)
    
    
if __name__=='__main__':
    en_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    cn_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    special_tokens_dict = {
    'additional_special_tokens': [IM_START, IM_END, BOS, EOS, PAD]
    }
    en_tokenizer.add_special_tokens(special_tokens_dict)
    cn_tokenizer.add_special_tokens(special_tokens_dict)
    
    en2cn = EN2CNDataset(en_tokenizer, cn_tokenizer)
    en, cn = en2cn.__getitem__(0)
    print('en',en)
    print('cn',cn)


    