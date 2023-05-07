import io
from argparse import ArgumentParser

from preprocessing.text import Tokenizer


def program(data_path: str,
            tokenizer_path: str,
            iterations: int,
            sigma: float):
    dataset = io.open(data_path, encoding='utf-8').read().strip().split("\n")

    tokenizer = Tokenizer(tokenizer_path)
    tokenizer.fit(dataset, iterations, sigma)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=1.5)

    args = parser.parse_args()

    if args.data_path is None or args.tokenizer_path is None:
        print("Missing Information")
    else:
        program(
            args.data_path,
            args.tokenizer_path,
            args.iterations,
            args.sigma
        )
