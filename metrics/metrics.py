from math import frexp
import torch

def accuracy(model, device, dataloader):
    '''
    accuracy = (tp + tn) / (p + n)
    '''
    model.eval()
    correct = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)

            _, predictions = torch.max(outputs, 1)

            correct += (predictions == y).sum().item()

    accuracy = correct / len(dataloader.dataset) * 100
    print(f"\tAccuracy = {accuracy:.2f}\n")
    return accuracy

def precision(model, device, dataloader):
    '''
    precision = tp / (tp + fp)
    '''
    model.eval()

    classes = model.fc2.out_features
    tp = torch.zeros(classes).to(device)
    fp = torch.zeros(classes).to(device)

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)

            _, predictions = torch.max(outputs, 1)

            for c in range(classes):
                tp[c] += ((predictions == c) & (y == c)).sum()
                fp[c] += ((predictions == c) & (y != c)).sum()

            precision = tp / (tp + fp)
            precision[torch.isnan(precision)] = 0

            precision = 100 * torch.mean(precision).item()

            print(f"\tPrecision = {precision:.2f}\n")
            return precision

def recall(model, device, dataloader):
    '''
    recall = tp / (tp + fn)
    '''
    model.eval()

    classes = model.fc2.out_features
    tp = torch.zeros(classes).to(device)
    fn = torch.zeros(classes).to(device)

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)

            _, predictions = torch.max(outputs, 1)

            for c in range(classes):
                tp[c] += ((predictions == c) & (y == c)).sum()
                fn[c] += ((predictions != c) & (y != c)).sum()

            recall = tp / (tp + fn)
            recall[torch.isnan(recall)] = 0
            recall = 100 * torch.mean(recall).item()

            print(f"\tRecall = {recall:.2f}\n")
            return recall
