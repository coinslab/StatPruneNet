def print_metrics(metrics: dict) -> None:
    print("Metrics")

    for metric, value in metrics.items():
        print(f'{metric}: {value:.4f}')