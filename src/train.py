from torch.utils.data import DataLoader
import torch
import os
from triplet_dataset import TripletDataset
from transforms import train_transform, test_transform
from siamese_network import SiamseNetwork
import time
from plotting import plot_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, roc_auc_score

# Constants
MODEL_SAVE_DIRECTORY = "models/train4/"
MODEL_FILENAME = "siamese_resnet18_v4.pth"
TRAIN_DATASET_DIRECTORY = "data/train/"
VALIDATION_DATASET_DIRECTORY = "data/val/"

# Loader
loader_batch_size = 32
loader_workers = 4

# Training
number_of_epochs = 20

def main(): 
    # Use GPU if available for training, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiamseNetwork()
    model.to(device)

    # Info to print before trainging
    print(f"Is cuda available: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    print(f"Device used by model: {next(model.parameters()).device}\n")

    #Generates a tiplet dataset
    train_dataset = TripletDataset(
        root=TRAIN_DATASET_DIRECTORY,
        transform=train_transform
    )

    validation_dataset = TripletDataset(
        root=VALIDATION_DATASET_DIRECTORY,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_batch_size,
        shuffle=True,
        num_workers=loader_workers,
        pin_memory=True,
        persistent_workers=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=loader_batch_size,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=True,
        persistent_workers=True
    )

    criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Losses
    train_losses = []
    val_losses = []

    # Metrics
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []

    # Training Loop
    best_val_loss = float("inf")
    for epoch in range(number_of_epochs):
        start = time.time()
        model.train()
        running_loss = 0

        for batch, (anchor, positive, negative) in enumerate(train_loader):
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)

            optimizer.zero_grad()
            anchor_embedding = model.encoder(anchor)
            positive_embedding = model.encoder(positive)
            negative_embedding = model.encoder(negative)

            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        # val_loss, val_acc = evaluate(model, validation_loader, criterion, device)
        val_loss, val_acc, val_precision, val_recall, val_f1, val_roc = evaluate(model, validation_loader, criterion, device)
        epoch_time = time.time() - start

        # Save values for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        accuracies.append(val_acc)
        precisions.append(val_precision)
        recalls.append(val_recall)
        f1_scores.append(val_f1)
        roc_aucs.append(val_roc)

        print(
            f"Epoch {epoch+1} took {epoch_time:.2f} seconds "
            f"train_loss={avg_train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"accuracy={val_acc:.4f} "
            f"precision={val_precision:.4f} "
            f"recall={val_recall:.4f} "
            f"f1={val_f1:.4f} "
            f"roc_auc={val_roc:.4f}"
        )
        # print(f"Epoch {epoch+1} Batch {batch} took {epoch_time:.2f} seconds: train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        ## Save based on VALIDATION loss, not training loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIRECTORY, MODEL_FILENAME))
            print("Model saved!")
    plot_metrics(MODEL_SAVE_DIRECTORY, train_losses, val_losses, accuracies, precisions, recalls, f1_scores, roc_aucs)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    y_true = []
    y_pred = []
    y_scores = []
    threshold = 0.6

    for anchor, positive, negative in loader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        a_emb = model.encoder(anchor)
        p_emb = model.encoder(positive)
        n_emb = model.encoder(negative)

        loss = criterion(a_emb, p_emb, n_emb)
        running_loss += loss.item()

        # ------------------------
        # Positive pairs
        # ------------------------
        pos_distance = torch.nn.functional.pairwise_distance(a_emb, p_emb)
        for distance in pos_distance:
            distance = distance.item()

            # Ground truth
            y_true.append(1)

            # Smaller distance means more similar.
            y_scores.append(-distance)

            # Prediction
            if distance < threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)

        # ------------------------
        # Negative pairs
        # ------------------------
        neg_distance = torch.nn.functional.pairwise_distance(a_emb, n_emb)
        for distance in neg_distance:
            distance = distance.item()
            y_true.append(0)
            y_scores.append(-distance)

            if distance < threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)

    avg_loss = running_loss / len(loader)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_scores)

    return (avg_loss, accuracy, precision, recall, f1, roc_auc)

if __name__ == "__main__":
    main()