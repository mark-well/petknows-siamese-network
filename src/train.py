from torch.utils.data import DataLoader
import torch
import os
from triplet_dataset import TripletDataset
from transforms import train_transform, test_transform
from siamese_network import SiamseNetwork

# Constants
MODEL_SAVE_DIRECTORY = "/content/drive/MyDrive/petknows/models"
MODEL_FILENAME = "siamese_resnet18_v2.pth"
TRAIN_DATASET_DIRECTORY = "/content/drive/MyDrive/petknows/dataset/train"
VALIDATION_DATASET_DIRECTORY = "/content/drive/MyDrive/petknows/dataset/val"

# Loader
loader_batch_size = 64
loader_workers = 4

# Training
number_of_epochs = 20

def main():
    # Use GPU if available for training, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    model = SiamseNetwork()
    model.to(device)

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

    # Training Loop
    best_val_loss = float("inf")
    for epoch in range(number_of_epochs):
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
        val_loss, val_acc = evaluate(model, validation_loader, criterion, device)
        print(f"Epoch {epoch+1} Batch {batch+1}: train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        ## Save based on VALIDATION loss, not training loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIRECTORY, MODEL_FILENAME))
            print("Model saved!")

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for anchor, positive, negative in loader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        a_emb = model.encoder(anchor)
        p_emb = model.encoder(positive)
        n_emb = model.encoder(negative)

        loss = criterion(a_emb, p_emb, n_emb)
        running_loss += loss.item()

        d_pos = torch.nn.functional.pairwise_distance(a_emb, p_emb)
        d_neg = torch.nn.functional.pairwise_distance(a_emb, n_emb)
        correct += (d_pos < d_neg).sum().item()
        total += anchor.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

if __name__ == "__main__":
    main()