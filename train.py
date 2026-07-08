from torch.utils.data import DataLoader
import torch

from datasets.triplet_dataset import TripletDataset
from datasets.transforms import train_transform
from siamese_network import SiamseNetwork

def main():
    model = SiamseNetwork()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = TripletDataset(
        root="dataset",
        transform=train_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training Loop
    number_of_epochs = 20
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

            print(
                f"Epoch {epoch+1}"
                f"Batcj {batch+1}/{len(loader)}"
                f"Loss: {loss.item():.4f}"
            )

        # Calculated average loss
        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1}: {avg_loss:.4f}" )
        
        # Save the model after training
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "siamese_resnet18.pth")
            print("Model saved!")


if __name__ == "__main__":
    main()