import os
from transformers import CLIPVisionModel, CLIPProcessor
import easyocr
import torch.cuda

clip_model_name = "openai/clip-vit-base-patch32"
CLIPVisionModel.from_pretrained(clip_model_name)
CLIPProcessor.from_pretrained(clip_model_name)

models_dir = "/app/models/easyocr"
os.makedirs(models_dir, exist_ok=True)
easyocr.Reader(["en"], gpu=("cuda" if torch.cuda.is_available() else "cpu"), model_storage_directory=models_dir, download_enabled=True)
