import torch

def metrics(model, device, dataloader):
    model.eval()
    correct = 0

    classes = model.fc2.out_features

    tp = torch.zeros(classes).to(device)
    fp = torch.zeros(classes).to(device)
    fn = torch.zeros(classes).to(device)

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)

            _, predictions = torch.max(outputs, 1)

            correct += (predictions == y).sum().item()

            for c in range(classes):
                tp[c] += ((predictions == c) & (y == c)).sum()
                fp[c] += ((predictions == c) & (y != c)).sum()
                fn[c] += ((predictions != c) & (y == c)).sum()

    accuracy = correct / len(dataloader.dataset) * 100

    precision = tp / (tp + fp)
    precision[torch.isnan(precision)] = 0
    precision = 100 * torch.mean(precision).item()

    recall = tp / (tp + fn)
    recall[torch.isnan(recall)] = 0
    recall = 100 * torch.mean(recall).item()

    return accuracy, precision, recall

# TODO
def print_results():
    pass
