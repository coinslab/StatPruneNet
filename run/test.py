from torch.utils.data import DataLoader
import torch
import torch.nn as nn

class Testing():
    def __init__(self, model, dataset, batch_size):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size

        self.test()

    def test(self):
        self.model.eval()
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        total = 0
        correct = 0

        with torch.no_grad():
            for input, label in dataloader:
                outputs = self.model(input)
                predictions = (outputs > 0.5).float()

                total += label.size(0)
                correct += (predictions == (label == 1).float().view(-1, 1)).sum().item()

        accuracy = correct / total
        print(f"Accuracy = {accuracy}")
