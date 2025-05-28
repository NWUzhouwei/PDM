import open_clip
import torch
import torch
import os
from torchvision.datasets import CIFAR100
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, token,preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai"
                                                                )
model=model.to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
texts = ["a photo of a airplane"]
text_tokens = tokenizer(texts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens).to(device)
print("Text features shape:", text_features.shape)  # (batch_size, 512)
print("Text features:", text_features)
