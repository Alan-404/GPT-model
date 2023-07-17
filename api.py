from fastapi import FastAPI
import torch
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from preprocessing.text import Tokenizer
import re
from trainer import GPTTrainer
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

origins = [
    "http://localhost:4200",
    "http://localhost:3000"
]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



tokenizer = Tokenizer("./tokenizer/dictionary.pkl", special_tokens=list(pd.read_json("./tokenizer/data.json").data.keys()))
end_token = tokenizer.dictionary.index("<end>")
model = GPTTrainer(len(tokenizer.dictionary), device=device, checkpoint='./saved_models/gpt.pt')
model.model.eval()

class Input(BaseModel):
    input: str

@app.post('/chat')
def chatbot(data:Input):
    digits = tokenizer.text_to_sequences([data.input], start_token=True, sep_token=True)
    len_seq = digits.shape[1]
    digits = torch.tensor(digits).to(device)
    result = model.predict(digits, 256, end_token)
    seq = []
    for word in result[0, len_seq:]:
        seq.append(tokenizer.dictionary[word.item()])
    seq = "".join(seq)
    print(seq)
    seq = re.sub("</w>", " ", seq)
    # seq = tokenizer.decode_special_tokens(seq)
    
    seq = seq.capitalize()
    return {'response': seq}