from torch.utils.data import DataLoader
import torch
import os
from src.triplet_dataset import TripletDataset
from src.transforms import train_transform
from src.siamese_network import SiamseNetwork

# Constants
MODEL_SAVE_DIRECTORY = "../models"
MODEL_FILENAME = "siamese_resnet18.pth"
DATASET_DIRECTORY = "../data/processed"

# Loader
loader_batch_size = 16
loader_workers = 0

# Training
number_of_epochs = 20

def main():
    # Use GPU if available for training, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiamseNetwork()
    model.to(device)

    #Generates a tiplet dataset
    dataset = TripletDataset(
        root=DATASET_DIRECTORY,
        transform=train_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=loader_batch_size,
        shuffle=True,
        num_workers=loader_workers
    )

    criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training Loop
    best_loss = float("inf")
    for epoch in range(number_of_epochs):
        model.train()
        running_loss = 0

        for batch, (anchor, positive, negative) in enumerate(loader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            anchor_embedding = model.encoder(anchor)
            positive_embedding = model.encoder(positive)
            negative_embedding = model.encoder(negative)

            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            print(f"Epoch {epoch+1} \nBatch {batch+1}/{len(loader)} \nLoss: {loss.item():.4f}")

        # Calculated average loss
        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}: {avg_loss:.4f}" )
        
        # Save the model after training
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIRECTORY, MODEL_FILENAME))
            print("Model saved!")


if __name__ == "__main__":
    main()