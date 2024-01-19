
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import PeftModel
 
# Save trained model
model_name = '../models/Taiwan-LLM-13B-v2.0-chat'

 
# Load the entire model on the GPU 0
device_map = 'auto'
 
new_model = "./output/DF-Taiwan-LLM-13B-v2.0"
 
# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
print("merge model and lora weights")
model = model.merge_and_unload()
 
output_merged_dir = "../models/final_DF_TW_llama2_13B"
os.makedirs(output_merged_dir, exist_ok=True)
print("save model")
model.save_pretrained(output_merged_dir, safe_serialization=True)
 
# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
 
print("save tokenizer")
tokenizer.save_pretrained(output_merged_dir)
