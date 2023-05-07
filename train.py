import torch
from trainer import GPTTrainer
from preprocessing.text import Tokenizer
import pickle
from model.utils.config import activations

from argparse import ArgumentParser


def load_data(path: str):
    with open(path, 'rb') as file:
        return pickle.load(file)

def program(tokenizer_path: str,
            n: int,
            d_model: int,
            heads: int,
            d_ff: int,
            dropout_rate: float,
            eps: float,
            activation: str,
            checkpoint: str,
            device: str,
            epochs: int,
            batch_size: int,
            mini_batch: int,
            learning_rate: float,
            validate_type: str,
            test_size: float,
            num_folds: int,
            data_path: str,
            with_mlflow: bool,
            mlflow_folder: str,
            experiment_name: str,
            run_id: str,
            run_name: str):
    tokenizer = Tokenizer(tokenizer_path)

    data = load_data(data_path)
    gpt = GPTTrainer(len(tokenizer.dictionary), n, d_model, heads, d_ff, dropout_rate, eps, activations[activation], device, checkpoint)

    data = torch.tensor(data)

    gpt.fit(data, 
            epochs=epochs, 
            batch_size=batch_size, 
            mini_batch=mini_batch, 
            learning_rate=learning_rate, 
            with_mlflow=with_mlflow,
            mlflow_folder=mlflow_folder,
            experiment_name=experiment_name,
            run_id=run_id,
            run_name=run_name,
            validate_type=validate_type,
            test_size=test_size,
            num_folds=num_folds
        )

if __name__ == "__main__":
    parser = ArgumentParser()

    # require
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--checkpoint", type=str)

    # model config
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--dropout_rate", type=int, default=0.1)
    parser.add_argument("--eps", type=int, default=0.02)
    parser.add_argument("--activation", type=str, default='gelu')

    # training config
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mini_batch", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.00003)

    # validation config
    parser.add_argument("--validate_type", type=str, default=None)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--num_folds", type=int, default=1)

    # visualization
    # - MLflow config
    parser.add_argument("--with_mlflow", type=bool, default=False)
    parser.add_argument("--mlflow_folder", type=str, default="./")
    parser.add_argument("--experiment_name", type=str, default="GPT Model")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="Version 1")


    args = parser.parse_args()
    if args.tokenizer_path is None or args.data_path is None or args.checkpoint is None:
        print("Missing Information")
    else:
        program(
            tokenizer_path=args.tokenizer_path,
            n=args.n,
            d_model=args.d_model,
            heads=args.heads,
            d_ff=args.d_ff,
            dropout_rate=args.dropout_rate,
            eps=args.eps,
            activation=args.activation,
            checkpoint=args.checkpoint,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            mini_batch=args.mini_batch,
            learning_rate=args.learning_rate,
            data_path=args.data_path,
            with_mlflow=args.with_mlflow,
            mlflow_folder=args.mlflow_folder,
            experiment_name=args.experiment_name,
            run_id=args.run_id,
            run_name=args.run_name,
            validate_type=args.validate_type,
            test_size=args.test_size,
            num_folds=args.num_folds
        )