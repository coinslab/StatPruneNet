from torchvision.transforms import transforms
from models.softmax_mlp import SoftmaxMLP
import torch.optim as optim
import torch.nn as nn
from experiment.pruning_experiment import PruningExperiment, Hyperparameters
import torchvision

if __name__ == '__main__':
    transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomCrop(28, padding=4),
        #transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=False)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=False)

    model = SoftmaxMLP(784, 10)
    learning_rate = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS(model.parameters(),
                            lr=learning_rate,
                            max_iter=200,
                            max_eval=250,
                            tolerance_grad=1e-20,
                            tolerance_change=1e-27,
                            history_size=100,
                            line_search_fn='strong_wolfe')

    hyperparams = Hyperparameters(epochs=50, gmin=1)

    experiment = PruningExperiment(model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            loss_fn=criterion,
            optimizer=optimizer,
            hyperparameters=hyperparams)
    
    experiment.experiment()
