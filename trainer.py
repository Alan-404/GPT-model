import torch
from torch import Tensor
import torch.nn.functional as F
from model.gpt import GPT
import torch.optim as optim
import torch.nn as nn
from typing import Union, Callable
from model.metric import BLEU
from torch.utils.data import TensorDataset, DataLoader
import os
import math
from sklearn.model_selection import train_test_split

class GPTTrainer:
    def __init__(self,
                 token_size: int,
                 n: int = 12,
                 d_model: int = 768,
                 heads: int = 12,
                 d_ff: int = 3072,
                 dropout_rate: float = 0.1,
                 eps: float = 0.02,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
                 device: str = "cpu",
                 checkpoint: str = None) -> None:
        if device is None:
            self.device = "cpu"
        elif device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = device
        self.model = GPT(token_size, n, d_model, heads, d_ff, dropout_rate, eps, activation)
        self.metric = BLEU()
        self.optimizer = optim.Adam(params=self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.loss_batch = 0.0
        self.loss_epoch = 0.0

        self.val_loss = 0.0
        self.val_accuracy = 0.0

        self.epoch = 0

        self.history = []

        self.pretrain_loss = 0.0

        self.with_mlflow = False
        self.with_tensorboard = False


        self.checkpoint = checkpoint
        self.model.to(device)
        if self.checkpoint is not None:
            self.load_checkpoint(self.checkpoint)

    def load_checkpoint(self, path: str):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.history = checkpoint['history']
            self.pretrain_loss = checkpoint['pretrain_loss']

    def save_checkpoint(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "history": self.history,
            'pretrain_loss': self.pretrain_loss
        }, path)

    def loss_function(self, outputs: Tensor, labels: Tensor) -> Tensor:
        batch_size = labels.size(0)
        loss = 0.0
        for batch in range(batch_size):
            loss += self.criterion(outputs[batch], labels[batch])
        loss = loss/batch_size
        return loss
    
    def build_dataset(self, data: Tensor, batch_size: int):
        dataset = TensorDataset(data)
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    def train_step(self, inputs: Tensor, labels: Tensor):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
 
        loss = self.loss_function(outputs, labels)
        loss.backward()
        self.optimizer.step()

        self.loss_batch += loss
        self.loss_epoch += loss

    def validate_step(self, inputs: Tensor, labels: Tensor):
        outputs = self.model(inputs)
        self.val_loss += self.loss_function(outputs, labels)
        _, predicts = torch.max(outputs, dim=-1)
        self.val_accuracy += self.metric.score(predicts, labels)


    def train_no_validate(self, dataset: Tensor, batch_size: int, epochs: int, mini_batch: int):
        if self.with_mlflow:
            import mlflow
        
        dataloader = self.build_dataset(dataset, batch_size=batch_size)

        for _ in range(epochs):
            count = 0
            for index, data in enumerate(dataloader, 0):
                inputs = data[0][:, :-1].to(self.device)
                labels = data[0][:, 1:].to(self.device)
                self.train_step(inputs, labels)
                count += 1
                if index%mini_batch == mini_batch-1 or index == len(dataloader) - 1:
                    print(f"Epoch: {self.epoch + 1} Batch: {index+1} Loss: {(self.loss_batch.item()/count):.4f}")
                    if self.with_mlflow:
                        mlflow.log_metric(f"Epoch {self.epoch + 1}", self.loss_batch.item()/count, index-count+1)
                    count = 0
                    self.loss_batch = 0.0
            print(f"Epoch: {self.epoch+1} Train Loss: {(self.loss_epoch.item()/len(dataloader)):.4f}")
            if self.with_mlflow:
                mlflow.log_metric(f"Train Loss", self.loss_epoch.item()/len(dataloader), step=self.epoch)
            self.history.append(self.loss_epoch.item()/len(dataloader))
            self.pretrain_loss = self.loss_epoch
            self.loss_epoch = 0.0
            self.epoch += 1
        print("Finished Tranining")

    def train_holdout_validate(self, dataset: Tensor, batch_size: int, epochs: int, mini_batch: int, test_size: float):
        if self.with_mlflow:
            import mlflow
        dataset_train, dataset_val = train_test_split(dataset, test_size=test_size)

        train_loader = self.build_dataset(dataset_train, batch_size=batch_size)
        val_loader = self.build_dataset(dataset_val, batch_size=batch_size)

        for _ in range(epochs):
            count = 0
            self.model.train()
            for index, data in enumerate(train_loader, 0):
                inputs = data[0][:, :-1].to(self.device)
                labels = data[0][:, 1:].to(self.device)
                self.train_step(inputs, labels)
                count += 1
                if index%mini_batch == mini_batch-1 or index == len(train_loader) - 1:
                    print(f"Epoch: {self.epoch + 1} Batch: {index+1} Loss: {(self.loss_batch.item()/count):.4f}")
                    if self.with_mlflow:
                        mlflow.log_metric(f"Epoch {self.epoch + 1}", self.loss_batch.item()/count, index-count+1)
                    count = 0
                    self.loss_batch = 0.0
            self.model.eval()
            for index, data in enumerate(val_loader, 0):
                inputs = data[0][:, :-1].to(self.device)
                labels = data[0][:, 1:].to(self.device)
                self.validate_step(inputs, labels)
            print(f"Epoch: {self.epoch+1} Train Loss: {(self.loss_epoch.item()/len(train_loader)):.4f}")
            print(f"Epoch: {self.epoch+1}, Validation Loss: {(self.val_loss.item()/len(val_loader)):4.f} BLEU Score: {(self.val_accuracy.item()/len(val_loader)):4f}")
            if self.with_mlflow:
                mlflow.log_metric(f"Train Loss", self.loss_epoch.item()/len(train_loader), step=self.epoch)
                mlflow.log_metric(f"Validation Loss", self.val_loss.item()/len(val_loader), step=self.epoch)
                mlflow.log_metric(f"BLEU Score", self.val_accuracy.item()/len(val_loader), step=self.epoch)
            self.history.append(self.loss_epoch.item()/len(train_loader))
            self.pretrain_loss = self.loss_epoch
            self.loss_epoch = 0.0
            self.epoch += 1
        print("Finished Tranining")

    def train_k_fold_validate(self, dataset: Tensor, batch_size: int, epochs: int, mini_batch: int, num_folds: int):
        if self.with_mlflow:
            import mlflow
        num_per_fold = math.ceil(dataset.size(0)/num_folds)

        for fold in list(range(num_folds)):
            start = fold*num_per_fold
            end = (fold+1)*num_per_fold
            val_dataset = dataset[start:end]
            train_dataset = torch.cat((dataset[:start], dataset[end:]), dim=0)
            
            train_dataloader = self.build_dataset(train_dataset, batch_size=batch_size)
            val_dataloader = self.build_dataset(val_dataset, batch_size=batch_size)
            total = len(train_dataloader)

            for _ in range(epochs):
                count = 0
                self.model.train()
                for index, data in enumerate(train_dataloader):
                    inputs = data[0][:, :-1].to(self.device)
                    labels = data[0][:, 1:].to(self.device)

                    self.train_step(inputs, labels)
                    count += 1
                    if index%mini_batch == (mini_batch-1) or index == total-1:
                        print(f"Epoch: {self.epoch+1} Batch: {index+1} Loss: {(self.loss_batch.item()/count):.4f}")
                        if self.with_mlflow:
                            mlflow.log_metric(f"Epoch {self.epoch+1}", self.loss_batch.item()/count, step=index - mini_batch + 1)
                        self.loss_batch = 0.0
                        count = 0
                self.model.eval()
                for index, data in enumerate(val_dataloader, 0):
                    inputs = data[0][:, :-1].to(self.device)
                    labels = data[0][:, 1:].to(self.device)
                    self.validate_step(inputs, labels)
                print(f"Epoch: {self.epoch+1} Train Loss: {(self.epoch_loss.item()/total):.4f}")
                print(f"Epoch: {self.epoch+1} Validation Loss: {(self.val_loss.item()/len(val_dataloader)):.4f} BLEU Score: {(self.val_accuracy.item()/len(val_dataloader)):.4f}")
                if self.with_mlflow:
                    mlflow.log_metric("Train Loss", self.epoch_loss.item()/total, step=self.epoch)
                    mlflow.log_metric("Validation Loss", self.val_loss.item()/len(val_dataloader), step=self.epoch)
                    mlflow.log_metric("BLEU Score", self.val_accuracy.item()/len(val_dataloader), step=self.epoch)
                self.epoch += 1
                self.epoch_loss = 0.0
                self.val_accuracy = 0.0
                self.val_loss = 0.0
        print("Finished Tranining")
    
    def fit(self, sequences: Tensor, epochs: int = 1, batch_size: int = 1, mini_batch: int = 1, learning_rate: float = 0.00003, **kwargs):    
        if kwargs['with_mlflow'] == True:
            import mlflow
            self.with_mlflow = True
            mlflow.set_tracking_uri(kwargs['mlflow_folder'])
            if kwargs['experiment_name'] is None:
                kwargs['experiment_name'] = "GPT Model"

            mlflow.set_experiment(kwargs['experiment_name'])

            if kwargs['run_id'] is None:
                if kwargs['run_name'] is None:
                    kwargs['run_name'] = "Version 1"
                mlflow.start_run(run_name=kwargs['run_name'])
            else:
                print(kwargs['run_id'])
                mlflow.start_run(run_id=kwargs['run_id'])

        for params in self.optimizer.param_groups:
            params['lr'] = learning_rate
        
        self.model.train()
        
        if kwargs['validate_type'] is None:
            self.train_no_validate(sequences, batch_size=batch_size, epochs=epochs, mini_batch=mini_batch)
        elif kwargs['validate_type'] == "holdout":
            self.train_holdout_validate(sequences, batch_size=batch_size, epochs=epochs, mini_batch=mini_batch, test_size=kwargs['test_size'])
        elif kwargs['validate_type'] == "kfold":
            self.train_k_fold_validate(sequences, batch_size=batch_size, epochs=epochs, mini_batch=mini_batch, num_folds=kwargs['num_folds'])
        if self.checkpoint is not None:
            self.save_checkpoint(self.checkpoint)

    def predict(self, seq: torch.Tensor, num_tokens: int, end_token: int):
        self.model.to(self.device)
        seq = seq.to(self.device)

        self.model.eval()

        for _ in range(num_tokens):
            output = self.model.inference(seq)

            predict = output[:, -1, :]
            _, token = torch.max(predict, dim=-1)

            if token == end_token:
                break

            seq = torch.cat([seq, token.unsqueeze(0)], dim=-1)

        return seq