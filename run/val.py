from torch.utils.data import DataLoader
import torch.nn as nn
import torch

class Validation():
    def __init__(self, model, dataset, criterion, batch_size):
        self.model = model
        self.dataset = dataset
        self.criterion = criterion
        self.batch_size = batch_size
        self.val_dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.avg_loss = self.val()

    def val(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for input, label in self.val_dataloader:
                y_hat = self.model(input)
                loss = self.criterion(y_hat, label)
                batch_size = label.size(0)
                total_loss += loss.item() * batch_size

        avg_loss = total_loss / len(self.dataset)
        print(f"Validation average loss = {avg_loss}")

        return avg_loss
