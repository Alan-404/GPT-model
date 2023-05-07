import bentoml
from preprocessing.text import Tokenizer
from trainer import GPTTrainer

from argparse import ArgumentParser


def program(name: str,
            tokenizer_path: str,
            checkpoint: str):
    tokenizer = Tokenizer(tokenizer_path)
    
    trainer = GPTTrainer(token_size=len(tokenizer.dictionary), checkpoint=checkpoint)

    print(bentoml.pytorch.save_model(name, trainer.model, signatures={"__call__": {'batchable': True, 'batch_dim':0}}))

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--name", type=str, default="gpt")
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--checkpoint", type=str)

    args = parser.parse_args()

    if args.tokenizer_path is None or args.checkpoint is None:
        print("Missing Information")
    else:
        program(
            args.name,
            args.tokenizer_path,
            args.checkpoint
        )