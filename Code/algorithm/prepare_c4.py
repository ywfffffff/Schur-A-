import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import random
import os

# 1. 配置
# 使用本地 Qwen3-30B-A3B 模型路径
TOKENIZER_ID = "/home/lab1008/data_disk_sdc/ywf/models/baidu/ERNIE-4.5-21B-A3B-PT" 
NUM_SAMPLES_EN = 512
NUM_SAMPLES_ZH = 0
NUM_SAMPLES_CHAT = 0
NUM_SAMPLES_CODE = 0
NUM_SAMPLES_MATH = 0
SEQ_LEN = 2048
OUTPUT_FILE = "/home/lab1008/data_disk_sdc/ywf/data/Calibrate/c4_ERNIE_512x2048.pt"

def collect_samples(dataset, tokenizer, target_count, seq_len, label, is_chat=False, is_code=False, is_math=False):
    samples = []
    count = 0
    skipped = 0
    print(f"开始采集 {label} 数据，目标: {target_count} 段 (严格 {seq_len} tokens)")
    
    it = iter(dataset)
    while count < target_count:
        try:
            item = next(it)
            if is_chat:
                if 'messages' in item:
                    text = ""
                    for msg in item['messages']:
                        text += f"{msg['role']}: {msg['content']}\n"
                else:
                    text = item.get('text', "")
            elif is_code:
                # flytech/python-codes-25k: instruction, input, output, text
                if 'text' in item and len(item['text']) > 10:
                    text = item['text']
                else:
                    text = f"{item.get('instruction', '')}\n{item.get('input', '')}\n{item.get('output', '')}"
            elif is_math:
                text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            else:
                text = item['text']
            
            # 先分词，不 padding，不截断，看原始长度
            tokens = tokenizer(text, truncation=False, return_tensors="pt", add_special_tokens=True)
            original_len = tokens.input_ids.shape[1]
            
            # 只保留长度 >= seq_len 的样本，然后截断
            if original_len >= seq_len:
                input_ids = tokens.input_ids[:, :seq_len]  # 截断到 seq_len
                samples.append(input_ids)
                count += 1
                if count % 10 == 0: 
                    print(f"[{label}] 已采集: {count}/{target_count} (跳过了 {skipped} 条短样本)")
            else:
                skipped += 1
        except StopIteration:
            print(f"[{label}] 数据集遍历完毕，只采集到 {count}/{target_count} 条")
            break
        except Exception as e:
            continue
    
    print(f"[{label}] 最终采集: {count} 条，跳过: {skipped} 条")
    return samples

def prepare_data():
    print(f"正在加载分词器: {TOKENIZER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    
    # 1. 加载英文数据集 (Wikipedia)
    print("正在加载 English 数据集 (c4)...")
    ds_en = load_dataset("allenai/c4", "en", split="train", streaming=True).shuffle(buffer_size=1000, seed=42)
    
    # # 2. 加载中文数据集 (Wikipedia zh)
    # print("正在加载 Chinese 数据集 (SkyPile)...")
    # ds_zh = load_dataset("Skywork/SkyPile-150B", split="train", streaming=True).shuffle(buffer_size=1000, seed=42)
    
    # # 3. 加载对话数据集 (UltraChat)
    # print("正在加载 Chat 数据集 (UltraChat)...")
    # ds_chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True).shuffle(buffer_size=1000, seed=42)

    # 4. 加载代码数据集 (flytech/python-codes-25k)
    # print("正在加载 Code 数据集 (flytech/python-codes-25k)...")
    # ds_code = load_dataset("flytech/python-codes-25k", split="train", streaming=True).shuffle(buffer_size=1000, seed=42)

    # # 5. 加载数学数据集 (GSM8K)
    # print("正在加载 Math 数据集 (GSM8K)...")
    # ds_math = load_dataset("gsm8k", "main", split="train", streaming=True).shuffle(buffer_size=1000, seed=42)

    samples_en = collect_samples(ds_en, tokenizer, NUM_SAMPLES_EN, SEQ_LEN, "EN")
    # samples_zh = collect_samples(ds_zh, tokenizer, NUM_SAMPLES_ZH, SEQ_LEN, "ZH")
    # samples_chat = collect_samples(ds_chat, tokenizer, NUM_SAMPLES_CHAT, SEQ_LEN, "CHAT", is_chat=False)
    # samples_code = collect_samples(ds_code, tokenizer, NUM_SAMPLES_CODE, SEQ_LEN, "CODE", is_code=True)
    # samples_math = collect_samples(ds_math, tokenizer, NUM_SAMPLES_MATH, SEQ_LEN, "MATH", is_math=True)
    
    # all_samples = samples_en + samples_zh + samples_chat + samples_code + samples_math
    all_samples = samples_en

    random.shuffle(all_samples)
    final_data = torch.cat(all_samples, dim=0)
    print(f"采集完成！最终形状: {final_data.shape}")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    torch.save(final_data, OUTPUT_FILE)
    print(f"成功！数据已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    random.seed(42)
    prepare_data()
