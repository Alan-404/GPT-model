import bentoml
import torch
from preprocessing.text import Tokenizer
import re
gpt_model = bentoml.pytorch.get("gpt:latest")

class GPTRunnable(bentoml.Runnable):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = bentoml.pytorch.load_model("gpt:latest")
        self.model.to(self.device)
        self.tokenizer = Tokenizer("./tokenizer/dictionary.pkl")
        self.num_token = 256
    
    @bentoml.Runnable.method(batchable=True)
    def generate(self, input_data: str) -> str:
        print(input_data)
        data = self.tokenizer.text_to_sequences([input_data], start_token=True, sep_token=True)
        data = torch.tensor(data).to(self.device)
        len_seq = data.size(1)
        result = self.model.predict(data, self.num_token, self.tokenizer.dictionary.index("<end>"))

        seq = []
        for word in result[0, len_seq:]:
            seq.append(self.tokenizer.dictionary[word.item()])
        seq = "".join(seq)
        seq = re.sub("</w>", " ", seq)
        seq = seq.capitalize()
        return seq
    
runner = bentoml.Runner(GPTRunnable, models=[gpt_model])
svc = bentoml.Service("chatbot", runners=[runner])

@svc.api(input=bentoml.io.JSON(), output=bentoml.io.Text())
def chatbot(input_text: str):
    result = runner.generate.run(input_text['data'])
    return result
    

