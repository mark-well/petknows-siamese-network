import os
import matplotlib.pyplot as plt

MODEL_SAVE_DIRECTORY = "models/"

def main():
    pass

def plot_metrics(save_dir, train_losses, val_losses, accuracies, precisions, recalls, f1_scores, roc_aucs):
    epochs = range(1, len(train_losses) + 1)

    # -------------------------
    # Plot Losses
    # -------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Training Loss")
    plt.plot(epochs, val_losses, marker="o", label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    plt.close()

    # -------------------------
    # Plot Classification Metrics
    # -------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracies, marker="o", label="Accuracy")
    plt.plot(epochs, precisions, marker="o", label="Precision")
    plt.plot(epochs, recalls, marker="o", label="Recall")
    plt.plot(epochs, f1_scores, marker="o", label="F1-score")
    plt.plot(epochs, roc_aucs, marker="o", label="ROC-AUC" )

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_plot.png"))
    plt.close()

if __name__ == "__main__":
    main()