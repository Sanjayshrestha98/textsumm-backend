
from fastapi import FastAPI, Request
# from fastapi import Request
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
# from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('./textsum_model')
tokenizer = T5Tokenizer.from_pretrained('./textsum_tokenizer')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) 

@app.get('/')
def index():
    return "Hello World"

@app.post("/summarize")
async def summarize(request:Request):
    req = await request.json()
    print(req)
    text = req['request']
    max_length = 6000

    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_Text = "summarize: " + preprocess_text
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

    summary_ids = model.generate(
        tokenized_text,
        num_beams=4,
        no_repeat_ngram_size=2,
        min_length=30,
        max_length=max_length,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"summary": summary}

 