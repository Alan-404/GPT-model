import pickle
from preprocessing.text import Tokenizer
import io
from argparse import ArgumentParser

def program(data_path: str,
            tokenizer_path: str,
            max_length: int,
            clean_path: str):
    tokenizer = Tokenizer(tokenizer_path)
    dataset = io.open(data_path, encoding='utf-8').read().strip().split("\n")

    clean_data = tokenizer.text_to_sequences(dataset, max_length=max_length, start_token=True, end_token=True)
    print(f"Data Shape: {clean_data.shape}")
    with open(clean_path, 'wb') as file:
        pickle.dump(clean_data, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--clean_path", type=str)
    
    args = parser.parse_args()

    if args.data_path is None or args.tokenizer_path is None or args.clean_path is None:
        print("Missing Information")
    else:
        program(
            args.data_path,
            args.tokenizer_path,
            args.max_length,
            args.clean_path
        )

