from fastapi import FastAPI, UploadFile
from PIL import Image
import io
from src.inference import get_embedding_async
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "").split(",")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],    
)

@app.get("/")
async def root():
    return {"message": "Server is running!"}

@app.post("/get_embedding")
async def get_embedding_endpoint(file: UploadFile) -> object:
    image_contents = await file.read()
    image = Image.open(io.BytesIO(image_contents))
    embedding = await get_embedding_async(image)
    return {
        "embedding": embedding.squeeze().cpu().tolist()
    }