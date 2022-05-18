
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi import FastAPI, Path

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

@app.get("/start")
def start():
    return {"Data": "Let's start chatting"}

@app.post("/respond/{sent}")
def respond(sent: str):
    new_user_input_ids = tokenizer.encode(sent + tokenizer.eos_token, return_tensors='pt')
    bot_respond_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(bot_respond_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return {"Data": response}