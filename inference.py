from models.transformer import Transformer
from configs.config import *
from transformers import BertTokenizer

def translate_sentence_en2zh(sentence, model, en_tokenizer, cn_tokenizer, device, max_len=50):
    # 1. 编码英文输入（加 BOS/EOS，如果训练时加了）
    sentence = BOS + sentence + EOS
    sentence_ids = en_tokenizer.encode(sentence, add_special_tokens=False)
    sentence_ids = torch.tensor([sentence_ids], dtype=torch.long).to(device)  # [1, seq_len]
    
    # 2. 初始化 decoder 输入（BOS）
    bos_id = cn_tokenizer.convert_tokens_to_ids(BOS)
    de_inputs_id = torch.tensor([[bos_id]], dtype=torch.long).to(device)
    
    # 3. 贪心解码
    for _ in range(max_len):
        with torch.no_grad():
            output = model(sentence_ids, de_inputs_id)  # [1, seq_len_dec, vocab_size]
        
        next_token_logits = output[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).item()
        
        if next_token_id == cn_tokenizer.convert_tokens_to_ids(EOS):
            break
        
        next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
        de_inputs_id = torch.cat([de_inputs_id, next_token_tensor], dim=1)
    
    # 4. 转换为中文字符串
    pred_tokens = [cn_tokenizer.convert_ids_to_tokens(i) for i in de_inputs_id[0].tolist()]
    pred_tokens = [tok for tok in pred_tokens if tok not in [BOS, EOS, PAD]]
    
    return "".join(pred_tokens)

if __name__== '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(enc_vocab_size,dec_vocab_size,q_k_size,v_size,emb_size,head,f_size,nblocks)
    checkpoint_path = ''
    checkpoints = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(deivce)
    model.eval()
    
    en_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    cn_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    special_tokens_dict = {
    'additional_special_tokens': [IM_START, IM_END, BOS, EOS, PAD]
    }
    en_tokenizer.add_special_tokens(special_tokens_dict)
    cn_tokenizer.add_special_tokens(special_tokens_dict)
