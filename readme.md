# ViT Medical Image Classification

This project implements fine-tuning of a **Vision Transformer (ViT)** on the **MedMNIST** dataset (specifically BloodMNIST).

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Configure settings in `config.yaml`.
3. Run training: `python train.py`
4. Run inference: `python inference.py --image path/to/image.png`

## Project Structure
- `src/`: Core logic for data and model definitions.
- `results/`: Contains performance logs and visualization curves.
- `checkpoints/`: Model weight storage.