#%%
import torch
from model.gpt import GPT
import pandas as pd
# %%
from preprocessing.text import Tokenizer
# %%
tokenizer = Tokenizer("./tokenizer/dictionary.pkl", special_tokens=list(pd.read_json("./tokenizer/data.json").data.keys()))
# %%
model = GPT(
    token_size=len(tokenizer.dictionary),
    n=12,
    d_model=768,
    heads=12,
    d_ff=3072,
    dropout_rate=0.1,
    eps=0.02,
    activation=torch.nn.functional.gelu
)
#%%
model = model.to('cuda')
# %%
model.load_state_dict(torch.load("./saved_models/gpt.pt")['model_state_dict'])
# %%
example = "hello"
# %%
digits = tokenizer.text_to_sequences([example], start_token=True, sep_token=True)
# %%
digits = torch.tensor(digits).to('cuda')
# %%
torch.jit.save(torch.jit.trace(model, digits), "gpt.pt")
# %%
import torchsummary
# %%
torchsummary.summary(model)
# %%
