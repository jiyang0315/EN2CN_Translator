from data.dataset import EN2CNDataset
from torch.utils.data import Dataset, DataLoader
from configs.config import pad_id, pad_id_cn
import torch
from transformers import BertTokenizer
from configs.config import *
import torch.optim as optim
from models.transformer import Transformer
from torch import nn
from tqdm import tqdm
import os


def my_collate(batch):
    max_en_len = 0
    max_cn_len = 0
    for en, cn in batch:
        max_en_len = max(max_en_len, len(en))
        max_cn_len = max(max_cn_len, len(cn))
    
    en_batch = []
    cn_batch = []
    for en, cn in batch:
        en_padded = en + [pad_id] * (max_en_len - len(en))
        cn_padded = cn + [pad_id_cn] * (max_cn_len - len(cn))
        en_batch.append(en_padded)
        cn_batch.append(cn_padded)
        
    en_tensor = torch.tensor(en_batch, dtype=torch.long)
    cn_tensor = torch.tensor(cn_batch, dtype=torch.long)
    return en_tensor, cn_tensor
    

if __name__=='__main__':
    # 加载数据
    en_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    cn_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    special_tokens_dict = {
    'additional_special_tokens': [IM_START, IM_END, BOS, EOS, PAD]
    }
    en_tokenizer.add_special_tokens(special_tokens_dict)
    cn_tokenizer.add_special_tokens(special_tokens_dict)

    en2cn_dataset_train = EN2CNDataset(en_tokenizer, cn_tokenizer, split='train')
    en2cn_dataset_val = EN2CNDataset(en_tokenizer, cn_tokenizer, split='val')
    en2cn_dataloader_train = DataLoader(en2cn_dataset_train, batch_size=128, shuffle=True, num_workers=4, collate_fn = my_collate)
    en2cn_dataloader_val = DataLoader(en2cn_dataset_val, batch_size=128, shuffle=True, num_workers=4, collate_fn = my_collate)
    
    # 创建网络
    model = Transformer(enc_vocab_size,dec_vocab_size,q_k_size,v_size,emb_size,head,f_size,nblocks)
    
    #创建损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    #开始训练
    model.train()
    NUM_EPOCHS = 10000
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        train_loss_total = 0
        for en_id, cn_id in en2cn_dataloader:
            # cn_id: [batch, seq_len]
            de_x = cn_id[:, :-1]      # decoder 输入
            target = cn_id[:, 1:]     # decoder 目标

            output = model(en_id, de_x)  # output shape: [batch, seq_len-1, vocab_size]

            # reshape 扁平化，CrossEntropyLoss 要求 [N, C] vs [N]
            loss = criterion(output.reshape(-1, output.size(-1)),
                            target.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
        # 打印 loss
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_train_loss.item():.4f}")

        # 每隔一定 epoch 保存一次模型
        if (epoch + 1) % 100 == 0 or (epoch + 1) == NUM_EPOCHS:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for en_id, cn_id in en2cn_dataloader:
                    de_x = cn_id[:, :-1]      # decoder 输入
                    target = cn_id[:, 1:]     # decoder 目标
                    de_x = cn_id[:, :-1]
                    output = model(en_id, de_x)
                    vloss = criterion(output.reshape(-1, output.size(-1)),
                                 target.reshape(-1))
                    val_loss_total += vloss.item()
                avg_val_loss = val_loss_total / len(en2cn_val_dataloader)
                print(f"--- Validation @ epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}")
                
            save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, save_path)
            print(f"Model saved to {save_path}")
            
            # 如果是更好的验证 loss，则保存 best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(save_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss
                }, best_path)
                print(f"New best model saved to {best_path}")

            model.train()
            