import os
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    plot_train_val()

def plot_train_val():
    report_id = 2601877
    report_file = os.path.join("reports", f"report_{report_id}.json")
    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)
    val_losses = report["val_losses"]
    train_losses = report["train_losses"]
    val_interval = report["val_interval"]
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(len(train_losses)), train_losses, label="Training Loss")
    # there is one val_loss every val_interval steps. So x values are 0, val_interval, 2*val_interval, ...
    plt.plot(np.arange(len(val_losses)) * val_interval, val_losses, label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss vs Steps")
    plt.legend()
    plt.savefig("train_val_loss.png")

def plot_lrs():
    report_ids = [2763469, 2765168, 2770519, 2772220]
    lrs = []
    val_arrs = []
    train_arrs = []
    val_interval = 0
    for report_id in report_ids:
        report_file = os.path.join("reports", f"report_{report_id}.json")
        with open(report_file, "r", encoding="utf-8") as f:
            report = json.load(f)
        lrs.append(report["optim_args"]["lr"])
        val_arrs.append(report["val_losses"])
        train_arrs.append(report["train_losses"])
        val_interval = report["val_interval"]
    plt.figure(figsize=(12, 5))
    for i, lr in enumerate(lrs):
        plt.plot(np.arange(len(train_arrs[i])), train_arrs[i], label=f"{lr}")
    # x goes from 0 to 5000
    plt.xlabel("Steps")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Steps")
    plt.legend()
    plt.savefig("training_loss.png")
    plt.figure(figsize=(12, 5))
    for i, lr in enumerate(lrs):
        plt.plot(np.arange(len(val_arrs[i])) * val_interval, val_arrs[i], label=f"{lr}")
    plt.xlabel("Steps")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs Steps")
    plt.legend()
    plt.savefig("validation_loss.png")

if __name__ == "__main__":
    main()