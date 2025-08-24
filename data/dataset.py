from torch.utils.data import Dataset
from transformers import BertTokenizer
from configs.config import MAX_SEQ_LEN

class EN2CNDataset(Dataset):
    # def __init__(self, en_tokenizer, cn_tokenizer):
    def __init__(self):
        super().__init__()
        self.en_list = []
        self.cn_list = []
        with open('data/train_data/train.en', 'r', encoding='utf-8') as fen, open('data/train_data/train.zh', 'r', encoding='utf-8') as fcn:
            for en in fen:
                self.en_list.append(en.strip())
            for cn in fcn:
                self.cn_list.append(cn.strip())
        assert len(self.en_list)==len(self.cn_list)
        
    def __len__(self):
        return len(self.en_list)
    
    def __getitem__(self, index):
        return self.en_list[index], self.cn_list[index]
    
    
if __name__=='__main__':
    en_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    cn_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    en2cn = EN2CNDataset()
    en, cn = en2cn.__getitem__(0)
    en_id = en_tokenizer.encode(en)
    en_id_special = en_tokenizer.encode(en)
    cn_id = cn_tokenizer.encode(cn)
    print(en_id_special)
    print(en_tokenizer.convert_ids_to_tokens(en_id_special))
    