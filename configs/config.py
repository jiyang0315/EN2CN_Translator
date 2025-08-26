from transformers import BertTokenizer
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IM_START = 'IM_START'
IM_END = 'IM_END'
BOS = '<sos>'
EOS = '<eos>'
PAD = '<pad>'
MAX_SEQ_LEN = 5000


en_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
cn_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
special_tokens_dict = {
'additional_special_tokens': [IM_START, IM_END, BOS, EOS, PAD]
}
en_tokenizer.add_special_tokens(special_tokens_dict)
cn_tokenizer.add_special_tokens(special_tokens_dict)

im_start_id = en_tokenizer.convert_tokens_to_ids(IM_START)
im_end_id   = en_tokenizer.convert_tokens_to_ids(IM_END)
bos_id      = en_tokenizer.convert_tokens_to_ids(BOS)
eos_id      = en_tokenizer.convert_tokens_to_ids(EOS)
pad_id      = en_tokenizer.convert_tokens_to_ids(PAD)
print("EN IDs:", im_start_id, im_end_id, bos_id, eos_id, pad_id)


im_start_id_cn = cn_tokenizer.convert_tokens_to_ids(IM_START)
im_end_id_cn   = cn_tokenizer.convert_tokens_to_ids(IM_END)
bos_id_cn      = cn_tokenizer.convert_tokens_to_ids(BOS)
eos_id_cn      = cn_tokenizer.convert_tokens_to_ids(EOS)
pad_id_cn      = cn_tokenizer.convert_tokens_to_ids(PAD)

print("CN IDs:", im_start_id_cn, im_end_id_cn, bos_id_cn, eos_id_cn, pad_id_cn)

enc_vocab_size = len(en_tokenizer)  # 英文词表大小（包含特殊 token）
dec_vocab_size = len(cn_tokenizer)  # 中文词表大小（包含特殊 token）
print("EN vocab size:", enc_vocab_size)
print("CN vocab size:", dec_vocab_size)

# 网络结构参数
emb_size = 128
q_k_size = 256
v_size = 512
f_size = 512
nblocks = 6
head = 8