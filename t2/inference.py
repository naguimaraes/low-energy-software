#!/usr/bin/env python3
"""
Run repeated inferences on the trained AlexNet model to measure GPU power consumption.
"""

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import time
from pathlib import Path

from classes import ConvolutionalAlexNet, FFTAlexNet, TorchAlexNet

def main():
    parser = argparse.ArgumentParser(description="Run AlexNet inference")
    parser.add_argument("--model", type=str, default="results/torch_11_100.pth", help="Path to saved model in the format <topology>_<kernel_size>_<epochs>.pth")
    parser.add_argument("--arch", choices=["convolutional", "fft", "torch"], default="torch",
                        help="Model architecture to instantiate before loading weights")
    parser.add_argument("--iterations", type=int, default=50, help="Number of full passes over the dataset to run (metrics computed on first pass only)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--data-root", type=str, default="dataset", help="Root directory containing CIFAR-100 data (expects 'cifar-100-python' inside)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model and load weights
    model_classes = {
        "convolutional": ConvolutionalAlexNet,
        "fft": FFTAlexNet,
        "torch": TorchAlexNet
    }
    
    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    first_kernel = checkpoint.get("first_kernel", 11)  # default to 11 if not saved
    
    model = model_classes[args.arch](num_classes=100, first_kernel=first_kernel).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Build CIFAR-100 test dataset and DataLoader
    normalize = transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.CIFAR100(root=args.data_root, train=False, download=False, transform=test_transform)
    if len(test_dataset) == 0:
        raise RuntimeError(f"No data found in '{args.data_root}'. Expected directory '{args.data_root}/cifar-100-python'.")

    pin_mem = device.type == "cuda"
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )

    print(f"Running inference over CIFAR-100 test set: {len(test_dataset)} images, batch_size={args.batch_size}...")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    processed = 0
    top1_correct = 0
    top5_correct = 0
    with torch.no_grad():
        for i in range(args.iterations):
            for b, (images, labels) in enumerate(test_loader, start=1):
                images = images.to(device, non_blocking=pin_mem)
                labels = labels.to(device, non_blocking=pin_mem)
                outputs = model(images)

                # Compute metrics only on the first pass to avoid skewing timing/energy
                if i == 0:
                    # Top-1 accuracy
                    pred1 = outputs.argmax(dim=1)
                    top1_correct += (pred1 == labels).sum().item()

                    # Top-5 accuracy
                    top5 = outputs.topk(5, dim=1).indices
                    top5_correct += top5.eq(labels.view(-1, 1)).any(dim=1).sum().item()

                processed += images.size(0)
                if b % 50 == 0:
                    print(f"Processed {processed} images ({b} batches in pass {i+1}/{args.iterations})")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"Inference completed: {processed} images in {elapsed:.2f} seconds.")
    # Report accuracy from the first pass
    if len(test_dataset) > 0:
        top1 = 100.0 * top1_correct / len(test_dataset)
        top5 = 100.0 * top5_correct / len(test_dataset)
        print(f"Top-1 Accuracy: {top1:.2f}% | Top-5 Accuracy: {top5:.2f}%")


if __name__ == "__main__":
    main()
