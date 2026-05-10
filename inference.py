import torch
from PIL import Image
from src.model import get_vit_model
from transformers import ViTImageProcessor

def run_inference(image_path, checkpoint_path, config):
    processor = ViTImageProcessor.from_pretrained(config['model']['name'])
    model = get_vit_model(config['model']['name'], config['num_classes'], 0.1, 0)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1)
    
    return prediction.item()