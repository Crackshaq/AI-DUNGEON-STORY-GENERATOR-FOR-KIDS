import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name="EleutherAI/gpt-neo-125M"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(torch.device("cpu"))  # Force CPU for Streamlit Cloud
    return tokenizer, model

def generate_story(prompt, tokenizer, model, max_length=200, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
        num_return_sequences=num_return_sequences
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
