# RCTNet-utils.py

from pathlib import Path
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.io import read_image
from enhance import EnhanceDataset, RCTNet  # Assuming enhance.py is in the same directory
import json

def enhance_images(image_path, checkpoint_path, config_path=None, batch_size=8, device=None):
    args = {
        'image': image_path,
        'checkpoint': checkpoint_path,
        'config': config_path,
        'batch_size': batch_size,
        'device': device
    }

    # Set path for checkpoints
    path = Path(args['image'])

    # Unless otherwise specified, model runs on CUDA if available
    if args['device'] is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args['device']

    # Initialize RCT model
    if args['config']:
        with open(args['config']) as fp:
            cfg = json.load(fp)  # load model configurations
        model = RCTNet(
            in_channels=cfg["in_channels"],
            hidden_dims=cfg["hidden_dims"],
            c_prime=cfg["c_prime"],
            epsilon=cfg["epsilon"],
            c_G=cfg["c_G"],
            n_G=cfg["n_G"],
            c_L=cfg["c_L"],
            n_L=cfg["n_L"],
            grid_size=cfg["grid_size"],
            device=device
        )
    else:
        model = RCTNet(device=device)

    # Move model to device selected
    model = model.to(device)

    # Load model's weights
    model.load_state_dict(torch.load(args['checkpoint'], map_location=torch.device(device)))
    model.eval()

    # Transform to convert torch.Tensor to PILImage
    transform = T.ToPILImage()

    if path.is_file() and path.suffix in ['.png', '.jpg', '.jpeg']:
        img = read_image(str(path)).float()
        img = torch.unsqueeze(img, 0).repeat(2, 1, 1, 1)

        with torch.no_grad():
            enhanced = torch.clamp(model(img)[0], max=255.0)
            enhanced_PIL = transform(enhanced / 255.0)

            save = path.with_stem(path.stem + "-enhanced")
            enhanced_PIL.save(save)
            exit()

    if path.is_dir():
        # Initialize dataloaders
        dataset = EnhanceDataset(path)
        dataloader = DataLoader(dataset, batch_size=args['batch_size'])

        for batch, x in enumerate(dataloader):
            with torch.no_grad():
                y = torch.clamp(model(x), max=255.0)

            for i, enhanced_img in enumerate(y):
                enhanced_img = transform(enhanced_img / 255.0)
                save = path / Path(dataset.images[i + batch*args['batch_size']])
                enhanced_img.save(save.with_stem(save.stem + "-enhanced"))

def enhance_images_from_script(image_path, checkpoint_path, config_path=None, batch_size=8, device=None):
    enhance_images(image_path, checkpoint_path, config_path, batch_size, device)

if __name__ == "__main__":
    pass

fin = enhance_images_from_script('/Users/albagir/Desktop/FinalProject/Logs2/crops/Prohibition - No turning right/0_896.jpg', '/Users/albagir/Desktop/FinalProject/RCT/checkpoint500.pt')