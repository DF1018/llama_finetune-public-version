import torch
from transformers import pipeline
import time

pipe = pipeline("text-generation", model="../models/final_DF_TW_llama2_13B", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
query = input("問題: ")
start = time.perf_counter()
messages = [
    {
        "role": "system",
        "content": "你是一個人工智慧助理",
    },
    {"role": "user", "content": query},
]

prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])

end = time.perf_counter()
print(f"執行時間：{(end - start)} 秒")