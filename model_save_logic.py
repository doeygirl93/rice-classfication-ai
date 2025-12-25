import torch
from pathlib import Path

def save_model(model, target_dir, model_name):
    target_dir_pth = Path(target_dir)
    target_dir_pth.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith('.pth'), "model name should end with a pth"

    model_save_path = target_dir_pth / model_name

    print(f"Saving model to {model_save_path}")
    torch.save(model.state_dict(), f=model_save_path)
