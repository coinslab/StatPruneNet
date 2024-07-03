from torchvision.transforms.transforms import ToTensor
from models.softmax_regression import SoftmaxRegression
import torch.optim as optim
import torch.nn as nn
from run.evaluate import Evaluate
import torchvision

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=ToTensor(), download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=ToTensor(), download=True)

model = SoftmaxRegression(784, 10)
learning_rate = 1
criterion = nn.CrossEntropyLoss()
optimizer = optim.LBFGS(model.parameters(), lr=learning_rate)
epochs = 5

Evaluate(model=model,
         train_dataset=train_dataset,
         test_dataset=test_dataset,
         criterion=criterion,
         optimizer=optimizer,
         epochs=epochs)
