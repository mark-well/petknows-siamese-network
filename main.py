import torch
import torch.nn as nn
from siamese_network import SiamseNetwork

model = SiamseNetwork()

img1 = torch.randn(1,3,640,640)
img2 = torch.randn(1,3,640,640)

emb1, emb2 = model(img1, img2)
similarity = nn.functional.cosine_similarity(emb1, emb2)
print(similarity)