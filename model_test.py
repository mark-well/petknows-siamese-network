import torch
from siamese_network import SiamseNetwork
from PIL import Image
from datasets.transforms import test_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiamseNetwork().to(device)
model.load_state_dict(torch.load("siamese_resnet18.pth", map_location=device))
model.eval()

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = test_transform(image)
    image = image.unsqueeze(0)

    return image

img1 = load_image("C:\\Users\\markw\\Downloads\\dog_faces_dataset\\train\\001161\\0003.jpg")
img2 = load_image("C:\\Users\\markw\\Downloads\\dog_faces_dataset\\train\\001173\\0007.jpg")

with torch.no_grad():
    emb1 = model.encoder(img1.to(device))
    emb2 = model.encoder(img2.to(device))

    similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
    print(similarity.item())