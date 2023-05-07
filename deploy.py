#%%
import bentoml
import torch
from model.gpt import GPTTrainer
from preprocessing.text import Tokenizer
# %%
tokenizer = Tokenizer("./tokenizer/dictionary.pkl")
# %%
trainer = GPTTrainer(token_size=len(tokenizer.dictionary), checkpoint="./saved_models/gpt.pt", device='cuda')
# %%
# %%
text = "what is deep learning?"
# %%
data = tokenizer.text_to_sequences([text], start_token=True, sep_token=True)
# %%
data = torch.tensor(data).to('cuda')
# %%
trainer.model.eval()
# %%
end_token = tokenizer.dictionary.index("<end>")
# %%
result = trainer.model.predict(data, 200, end_token)
# %%
data.device
# %%
result
# %%
for item in result[0]:
    print(tokenizer.dictionary[item.item()])
# %%
torch.cuda.empty_cache()
# %%
