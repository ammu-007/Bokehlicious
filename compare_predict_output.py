import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchmetrics.functional.image import learned_perceptual_image_patch_similarity as lpips

def main():
    # Paths
    predicted_path = r'E:\antigravity_projects\Bokehlicious\output\net_small_f2.8_collie.jpg'
    gt_path = r'E:\antigravity_projects\Bokehlicious\examples\collie_gt.jpg'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"\nComparing predicted output with ground truth...")
    print(f"Predicted: {predicted_path}")
    print(f"Ground Truth: {gt_path}")
    
    # Load images
    predicted = Image.open(predicted_path).convert('RGB')
    gt = Image.open(gt_path).convert('RGB')
    
    # Convert to tensors
    predicted_tensor = to_tensor(predicted).unsqueeze(0).to(device)
    gt_tensor = to_tensor(gt).unsqueeze(0).to(device)
    
    print(f"\nPredicted image size: {predicted.size}")
    print(f"Ground truth image size: {gt.size}")
    
    # Calculate metrics
    with torch.no_grad():
        psnr_val = psnr(predicted_tensor, gt_tensor, data_range=1.0).item()
        ssim_val = ssim(predicted_tensor, gt_tensor, data_range=1.0).item()
        lpips_val = lpips(predicted_tensor, gt_tensor, normalize=True).item()
    
    print(f"\n{'='*50}")
    print(f"Metrics (predict.py output vs ground truth):")
    print(f"{'='*50}")
    print(f"PSNR:  {psnr_val:.3f}")
    print(f"SSIM:  {ssim_val:.4f}")
    print(f"LPIPS: {lpips_val:.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
