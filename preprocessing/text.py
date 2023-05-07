import re
import pickle
import os
import numpy as np
import datetime

class Cleaner:
    def __init__(self, puncs: list = r"([:./,?!@#$%^&=`~*\(\)\[\]\"\'\-\\])") -> None:
        self.puncs = puncs
    def clean(self, seq: str):
        seq = re.sub(self.puncs, r" \1 ", seq)
        seq = seq.strip()
        seq = re.sub("\s\s+", " ", seq)
        seq = seq.lower()
        return seq
class Replacer:
    def __init__(self) -> None:
        self.replace_patterns = [
            (r"won\'t", 'will not'),
            (r"can't", 'cannot'),
            (r"'ll", " will"),
            (r"n\'t", "not"),
            (r"don\'t", "do not"),
            (r"doesn\'t", "does not"),
            (r"'ve", " have"),
            (r"i\'m", 'i am'),
            (r"'re", " are"),
            (r"wasn't", "was not"),
            (r"weren't", "were not"),
            (r"\'d", " would"),
            (r"\'s", " 's")
        ]


    def replace(self, sequence: list) -> list:
        for (pattern, repl) in self.replace_patterns:
            sequence = re.sub(pattern, repl, sequence)

        return sequence


class Tokenizer:
    def __init__(self, pretrained: str = None, 
                 special_tokens: list = ["<pad>", "<oov>", "<start>", "<end>", "<delim>", "<sep>", "<new_line>", "<get_time>", "<owner_name>", "<chatbot_name>", "<open_camera>", "<close_camera>", "<list>"]) -> None:
        self.cleaner = Cleaner()
        self.replacer = Replacer()
        self.vocab_dict = dict()
        self.dictionary = []

        self.special_tokens = special_tokens
        self.original_size = 0
        self.epoch = 0

        self.pretrained = pretrained
        if self.pretrained is not None:
            self.load_tokenizer(self.pretrained)
    
    def save_tokenizer(self, path: str):
        obj = {
            "dictionary": self.dictionary,
            "vocabulary": self.vocab_dict,
            "original_size": self.original_size,
            'epoch': self.epoch
        }
        with open(path, 'wb') as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_tokenizer(self, path: str):
        if os.path.exists(path):
            with open(path, 'rb') as file:
                data = pickle.load(file)
            self.dictionary = data['dictionary']
            self.vocab_dict = data['vocabulary']
            self.original_size = data['original_size']
            self.epoch = data['epoch']

    def cal_total_vocab(self, data: list):
        dictionary = []
        for item in data:
            seq = self.cleaner.clean(item)
            words = seq.split(" ")
            for word in words:
                if word not in dictionary:
                    dictionary.append(word)
        return len(dictionary)

    
    def __init_vocab_dict(self, data: list):
        vocab_dict = dict()
        dictionary = []
        for seq in data:
            seq = self.cleaner.clean(seq)
            seq = self.replacer.replace(seq)
            words = seq.split(" ")
            for word in words:
                if word not in dictionary:
                    dictionary.append(word)
                temp = []
                if word in self.special_tokens or word in self.cleaner.puncs:
                    temp.append(word)
                else:
                    for char in word:
                        temp.append(char)
                    temp.append("</w>")
                if tuple(temp) not in vocab_dict:
                    vocab_dict[tuple(temp)] = 1
                else:
                    vocab_dict[tuple(temp)] += 1

        return vocab_dict, len(dictionary)
    
    def create_dictionary(self, dictionary: dict):
        tokens = []
        for token in self.special_tokens:
            tokens.append(token)
        for item in dictionary:
            for char in item:
                if char not in tokens:
                    tokens.append(char)
        return tokens
    
    def create_pair(self, vocab_dict: dict):
        pair = dict()
        for vocab in vocab_dict:
            for index in range(len(vocab)-1):
                if (vocab[index], vocab[index+1]) not in pair:
                    pair[(vocab[index], vocab[index+1])] = 1
                else:
                    pair[(vocab[index], vocab[index+1])] += 1
        return pair
    
    def decode_special_tokens(self, seq: str):
        seq = re.sub("<get_time>", str(datetime.datetime.now()), seq)
        seq = re.sub("<chatbot_name>", "Lily", seq)
        seq = re.sub("<owner_name>", "Tri ", seq)
        seq = re.sub(" , ", ",", seq)
        seq = re.sub("<new_line>", "\n", seq)
        return seq

    def fit(self, data: list, max_iterations: int = 10, sigma: float = 2.0):
        if self.original_size == 0 and len(self.dictionary) == 0:
            self.vocab_dict, self.original_size = self.__init_vocab_dict(data)
            self.dictionary = self.create_dictionary(self.vocab_dict)
        print(f"Original Dictionary Size: {self.original_size}")
        print("========== Training Tokenizer ============")
        for _ in range(max_iterations):
            pairs = self.create_pair(self.vocab_dict)
            max_item = max(pairs, key=lambda k: pairs[k])
            temp_dict = dict()
            for vocab in self.vocab_dict:
                temp = []
                flag = False
                for index in range(len(vocab)):
                    if flag is True:
                        flag = False
                        continue
                    if vocab[index] != max_item[0]:
                        temp.append(vocab[index])
                    else:
                        if index == len(vocab)-1:
                            temp.append(vocab[index])
                            continue
                        if vocab[index + 1] == max_item[1]:
                            temp.append(vocab[index] + vocab[index+1])
                            flag = True
                        else:
                            temp.append(vocab[index])
                temp_dict[tuple(temp)] = self.vocab_dict[vocab]
            
            self.vocab_dict = temp_dict
            self.dictionary = self.create_dictionary(self.vocab_dict)
            print(f"Epoch {self.epoch+1} Dictionary Size: {len(self.dictionary)}")
            self.epoch += 1
            if len(self.dictionary) >= int(self.original_size/sigma) and len(self.dictionary) < self.original_size:
                break
        if self.pretrained is not None:
            self.save_tokenizer(self.pretrained)
    
    def find(self, word: str, special_token: bool = False):
        text = [*word]
        if special_token == False:
            text += ["</w>"]
        else:
            return [self.dictionary.index(word)]
        embedding = []
        mixed = 0
        for index in range(len(text)):
            if mixed > index:
                continue
            if mixed >= len(text):
                break
            subseq = len(text[index:])
            for i in range(subseq):
                pattern = "".join(text[index:subseq-i + index])
                if pattern in self.dictionary:
                    embedding.append(self.dictionary.index(pattern))
                    mixed += len(text[index:subseq - i + index])
                    break
        return embedding
    
    def get_special_tokens(self, token):
        return self.dictionary.index(token)
    
    def padding_sequence(self, sequence, padding: str, maxlen: int) -> np.ndarray:
        delta = maxlen - len(sequence)
        zeros = np.zeros(delta, dtype=np.int64)

        if padding.strip().lower() == 'post':
            return np.concatenate((sequence, zeros), axis=0)
        elif padding.strip().lower() == 'pre':
            return np.concatenate((zeros, sequence), axis=0)

    def truncating_sequence(self, sequence, truncating: str, maxlen: int) -> np.ndarray:
        if truncating.strip().lower() == 'post':
            return sequence[0:maxlen]
        elif truncating.strip().lower() == 'pre':
            delta = sequence.shape[0] - maxlen
            return sequence[delta: len(sequence)]

    def pad_sequences(self, sequences: list, maxlen: int, padding: str = 'post', truncating: str = 'post') -> np.ndarray:
        result = []
        for _, sequence in enumerate(sequences):
            delta = sequence.shape[0] - maxlen
            if delta < 0:
                sequence = self.padding_sequence(sequence, padding, maxlen)
            elif delta > 0:
                sequence = self.truncating_sequence(sequence, truncating, maxlen)
            result.append(sequence)
        
        return np.array(result)

    def text_to_sequences(self, data: list, max_length: int = None, start_token: bool = False, end_token: bool = False, sep_token: bool = False):
        digits = []
        maxlen = 0
        for seq in data:
            seq = self.cleaner.clean(seq)
            seq = self.replacer.replace(seq)
            words = seq.split(" ")
            temp = []
            if start_token:
                temp += [self.get_special_tokens("<start>")]
            for word in words:
                special_token = word in self.special_tokens
                digit_word = self.find(word, special_token)
                temp += digit_word
            if sep_token:
                temp += [self.get_special_tokens("<sep>")]
            if end_token:
                temp += [self.get_special_tokens("<end>")]
            if maxlen < len(temp):
                maxlen = len(temp)
            digits.append(np.array(temp))
        if max_length is None:
            padded_data = self.pad_sequences(digits, maxlen=maxlen)
        else:
            padded_data = self.pad_sequences(digits, max_length)
        return padded_data

