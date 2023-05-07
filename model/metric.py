import torch
import numpy as np

class BLEU:
    def __init__(self, n_grams: int = 4, uniform_weights: tuple | list = [0.25, 0.25, 0.25, 0.25]):
        if n_grams != len(uniform_weights):
            print("Confliced weights of grams")
            return
        self.n_grams = n_grams
        self.uniform_weights = uniform_weights

    def precision_grams(self, output: torch.Tensor, label: torch.Tensor, grams: int):
        current = 0
        length = label.size(0)
        correct = 0

        while True:
            if torch.equal(input=output[current:current+grams], other=label[current:current+grams]):
                correct += 1
            if current+grams == length:
                break
            current += 1
        result = correct/(length-grams+1)
        return result

    def geometric_average_precision_scores(self, output: torch.Tensor, label: torch.Tensor):
        gaps = 1
        for n in range(self.n_grams):
            precision_score = self.precision_grams(output=output, label=label, grams=n+1)
            gaps = gaps*np.power(precision_score, self.uniform_weights[n])

        return gaps

    def brevity_penalty(self, output: torch.Tensor, label: torch.Tensor):
        length_ref = (label != 0).sum()
        length_pred = (output != 0).sum()
        if length_pred >= length_ref:
            return torch.exp(1 - (length_ref/length_pred))
        return 1

    def score(self, outputs: torch.Tensor, labels: torch.Tensor):
        total_score = 0.0
        batch_size = labels.size(0)
        for index in range(batch_size):
            gaps = self.geometric_average_precision_scores(output=outputs[index], label=labels[index])
            penalty = self.brevity_penalty(output=outputs[index], label=labels[index])
            
            batch_score = gaps*penalty
            total_score += batch_score

        return total_score/batch_size