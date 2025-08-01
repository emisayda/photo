import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from model.u2net import U2NET  # Make sure model/ has __init__.py

def load_model(model_path='saved_models/u2net.pth'):
    model = U2NET(3, 1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict(model, image_path):
    image = Image.open(image_path).convert('RGB')
    original_size = image.size

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        d1, *_ = model(input_tensor)
    pred = d1[0][0].cpu().numpy()
    mask = (pred - pred.min()) / (pred.max() - pred.min())
    mask = (mask * 255).astype(np.uint8)

    mask = Image.fromarray(mask).resize(original_size).convert('L')
    image = image.convert("RGBA")
    image.putalpha(mask)

    return image
