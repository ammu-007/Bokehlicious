from torch import load, no_grad
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
import yaml

from method.model import Bokehlicious
from method.config import bokehlicious_size_builder

from dataset.util import load_image
from util.parser import get_predict_parser

if __name__ == "__main__":
    parser = get_predict_parser()
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)

    exp_name    = config_data['experiment']['name']
    size        = config_data['model']['size']
    device      = config_data['model']['device']
    
    img_path    = config_data['inference']['img_path']
    av          = float(config_data['inference']['av'])
    max_dim     = int(config_data['inference']['max_dim'])
    
    ckpt_dir    = Path(config_data['logging']['checkpoint_dir']) / exp_name
    out_path    = Path(config_data['inference']['out_path']) / exp_name
    out_path.mkdir(parents=True, exist_ok=True)

    config = bokehlicious_size_builder(size)

    model = Bokehlicious(**config)

    print(f"Initialized {size} Bokehlicious model on {device} (Exp: {exp_name})")

    checkpoint = ckpt_dir / f"{size}_best.pt"
    if not checkpoint.exists():
        checkpoint = ckpt_dir / f"{size}.pt"


    state_dict = load(checkpoint)

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print(f"Loaded weights from {checkpoint}")

    net_input = load_image(img_path, target_av=av, max_dim=max_dim, device=device)

    with no_grad():
        out = model(**net_input)

    print(f"Rendered {img_path} at f{av}")

    out_img = to_pil_image(out.cpu().detach().squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy())

    output_file = out_path / f'net_{size}_f{av}_{Path(img_path).name}'
    out_img.save(output_file)

    print(f"Saved result to {output_file}")
