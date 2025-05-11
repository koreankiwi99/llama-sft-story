import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and tokenizer
base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct" 
finetuned_repo = "kiwi1229/llama3.1-8b-moviebot-lora"     

tokenizer = AutoTokenizer.from_pretrained(finetuned_repo, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, finetuned_repo)

# Run inference
prompt = "Can you recommend me a fun superhero movie with family themes?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))