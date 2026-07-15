import torch
from src.siamese_network import SiamseNetwork
from PIL import Image
from src.transforms import test_transform
import asyncio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiamseNetwork().to(device)
model.load_state_dict(torch.load("models/train3/siamese_resnet18_v3.pth", map_location=device))
model.eval()

def load_image(image: Image.Image):
    image = image.convert("RGB")
    image = test_transform(image)
    image = image.unsqueeze(0)

    return image

def get_embedding(image: Image.Image):
    img = load_image(image)
    with torch.no_grad():
        return model.get_embedding(img.to(device))
    
async def get_embedding_async(image: Image.Image):
    return await asyncio.to_thread(get_embedding, image)