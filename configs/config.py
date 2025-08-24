# IM_START = 'IM_START'
# IM_END = 'IM_END'
# BOS = '<sos>'
# EOS = '<eos>'
# PAD = '<pad>'
MAX_SEQ_LEN = 5000


from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 如果你想用自定义 PAD token，需要先添加
# tokenizer.add_special_tokens({'pad_token': PAD})

PAD_IDX = tokenizer.pad_token_id
print('PAD_IDX',PAD_IDX)