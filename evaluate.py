from os import makedirs
from pathlib import Path
import yaml

from torch import load, no_grad, clamp, Tensor
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from dataset.loader import RealBokeh, EBB
from dataset.util import Mode
from method.config import bokehlicious_size_builder
from method.model import Bokehlicious
from util.parser import get_eval_parser

from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchmetrics.functional.image import learned_perceptual_image_patch_similarity as lpips

def append_av(av_dict, key, value):
    if key in av_dict:
        av_dict[key].append(value)
    else:
        av_dict[key] = [value]

def preprocess_batch(batch):
    for k, v in batch.items():
        if isinstance(v, Tensor):
            batch[k] = v.unsqueeze(0).cuda()
    return batch

if __name__ == "__main__":
    parser = get_eval_parser()
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)

    exp_name     = config_data['experiment']['name']
    size         = config_data['model']['size']
    device       = config_data['model']['device']
    
    dataset_name = config_data['inference']['dataset']
    save_outputs = config_data['inference']['save_outputs']
    img_format   = config_data['inference']['image_format']
    
    ckpt_dir     = Path(config_data['logging']['checkpoint_dir']) / exp_name
    out_path     = Path(config_data['inference']['out_path']) / exp_name

    config = bokehlicious_size_builder(f"{size}{'_bin' if (dataset_name == 'RealBokeh_bin') else ''}")

    model = Bokehlicious(**config)

    print(f"Initialized {size} Bokehlicious model on {device} (Exp: {exp_name})")

    checkpoint = ckpt_dir / f"{size}_best.pt"
    if not checkpoint.exists():
        checkpoint = ckpt_dir / f"{size}{'_bin' if dataset_name == 'RealBokeh_bin' else '' if dataset_name == 'RealBokeh' else f'_{dataset_name}'}.pt"

    state_dict = load(checkpoint, map_location=device)

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print(f"Loaded weights from {checkpoint}")

    if dataset_name == "RealBokeh":
        dataloader = RealBokeh(data_path="./dataset/RealBokeh_3MP", mode=Mode.TEST, device=device)
    elif dataset_name == "RealBokeh_bin":
        dataloader = RealBokeh(data_path="./dataset/RealBokeh_3MP", mode=Mode.TEST, binary_bokeh=True, device=device)
    elif dataset_name == "EBB400":
        dataloader = EBB(data_path="./dataset/EBB400", mode=Mode.VAL, device=device)
    elif dataset_name == "EBB_Val294":
        dataloader = EBB(data_path="./dataset/EBB_Val294", mode=Mode.VAL, device=device)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Initialized {dataset_name} dataloader")

    print(f"Calculating metrics for RealBokeh {size} on {dataset_name} dataset...")

    if save_outputs:
        out_ds_path = out_path / dataset_name
        out_ds_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving outputs to {out_ds_path}!")
    else:
        print("Not saving outputs! (Set save_outputs: true in config to save)")

    lpips_vals = []
    ssim_vals = []
    psnr_vals = []

    lpips_avs = {}
    ssim_avs = {}
    psnr_avs = {}

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        with no_grad():
            output = clamp(model(**preprocess_batch(batch)), 0, 1)

            psnr_val = psnr(output, batch['target'], data_range=1.0).item()
            ssim_val = ssim(output, batch['target'], data_range=1.0).item()
            lpips_val = lpips(output, batch['target'], normalize=True).item()

        psnr_vals.append(psnr_val)
        ssim_vals.append(ssim_val)
        lpips_vals.append(lpips_val)

        if dataset_name == "RealBokeh":
            av = batch['image_name'][0].split("_")[1]
            append_av(lpips_avs, av, lpips_val)
            append_av(ssim_avs, av, ssim_val)
            append_av(psnr_avs, av, psnr_val)

        if save_outputs:
            to_pil_image(output.squeeze(0).cpu()).save(out_ds_path / f"{batch['image_name'][0]}.{img_format}")

    print(f"Results for Bokehlicious {size} on {dataset_name}")
    print(f"Mean PSNR: {sum(psnr_vals) / len(psnr_vals):.3f}")
    print(f"Mean SSIM: {sum(ssim_vals) / len(ssim_vals):.4f}")
    print(f"Mean LPIPS: {sum(lpips_vals) / len(lpips_vals):.4f}")

    if dataset_name == "RealBokeh":
        print("------------------------------------------------------")
        for key in sorted(lpips_avs, key=lambda x: float(x.split("f")[-1])):
            print(f"Mean PSNR {key}: {sum(psnr_avs[key]) / len(psnr_avs[key]):.4f}")
            print(f"Mean SSIM {key}: {sum(ssim_avs[key]) / len(ssim_avs[key]):.4f}")
            print(f"Mean LPIPS {key}: {sum(lpips_avs[key]) / len(lpips_avs[key]):.3f}")
            print("------------------------------------------------------")
