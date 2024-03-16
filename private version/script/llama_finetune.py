# 匯入必要的模組和套件
import os
import torch
from torch import cuda, bfloat16
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
print(f"Model loaded on {device}")


model_path = '../models/final_DF_TW_llama2_13B_1.0'
new_model = "./output/DF_finetune_13B_0115"
#dataset_name = "./train.jsonl"

################################################################################
# Quantized LLMs with Low-Rank Adapters (QLoRA) parameters
################################################################################
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters 輕量級封裝，專門用於CUDA自定義函數，特別是8位優化器、矩陣乘法和量化
################################################################################
use_4bit = True
bnb_4bit_compute_dtype = "bfloat16" # float16 or bfloat16
bnb_4bit_quant_type = "nf4" # fp4 or nf4
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################
output_dir = "./output/results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 1
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 5
logging_steps = 5

################################################################################
# Supervised finetuning (SFT) parameters
################################################################################
max_seq_length = 512
packing = False
device_map = "auto" #{"": 0} or "auto"

# 讀取資料集
train_dataset = load_dataset('json', data_files='./data/train-data.jsonl', split="train")  # 從JSON文件中載入訓練數據集
valid_dataset = load_dataset('json', data_files='./data/eval-data.jsonl', split="train")  # 從JSON文件中載入驗證數據集

# 對數據集進行前處理，將提示和回應組合成文本對
train_dataset = train_dataset.map(lambda examples: {'text': [prompt + response for prompt, response in zip(examples['prompt'], examples['response'])]}, batched=True)
valid_dataset = valid_dataset.map(lambda examples: {'text': [prompt + response for prompt, response in zip(examples['prompt'], examples['response'])]}, batched=True)


# 定義位元和字節量化的相關配置
# dataset_name = "mlabonne/guanaco-llama2-1k"
# dataset = load_dataset(dataset_name, split="train")

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# 檢查 GPU 是否與 bfloat16 相容
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# 從預訓練模型中載入自動生成模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1



# 載入與模型對應的分詞器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# 定義 Prompt Engineering Fine-Tuning （PEFT）的相關設定
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# 設置訓練參數
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard", #"all"
    evaluation_strategy="steps",
    eval_steps=5  # 每5部驗證
)

# 使用 SFTTrainer 進行監督式微調訓練
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset, # 在這裡傳入驗證數據集
    eval_dataset=valid_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# 開始訓練模型
trainer.train()

# 儲存微調後的模型
trainer.model.save_pretrained(new_model)