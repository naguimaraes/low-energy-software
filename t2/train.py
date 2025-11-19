# train.py
import os
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from classes import ConvolutionalAlexNet, FFTAlexNet, TorchAlexNet

def get_dataloaders(batch_size=128, num_workers=4):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    trainset = torchvision.datasets.CIFAR100(root='./dataset', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./dataset', train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return trainloader, testloader

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    times = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        t0 = time.perf_counter()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t1 = time.perf_counter()
        times.append(t1 - t0)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    avg_time = sum(times) / len(times) if len(times) else 0.0
    return total_loss / total, 100. * correct / total, avg_time

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    top1_correct = 0
    top5_correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            
            # Top-1 accuracy
            _, predicted = outputs.max(1)
            top1_correct += predicted.eq(targets).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
            top5_correct += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).any(dim=1).sum().item()
            
            total += targets.size(0)
    
    avg_loss = total_loss / total
    top1_acc = 100. * top1_correct / total
    top5_acc = 100. * top5_correct / total
    return avg_loss, top1_acc, top5_acc

def build_model(model_name, first_kernel, num_classes, device):
    if model_name == "convolutional":
        model = ConvolutionalAlexNet(num_classes=num_classes, first_kernel=first_kernel)
    elif model_name == "fft":
        model = FFTAlexNet(num_classes=num_classes, first_kernel=first_kernel)
    elif model_name == "torch":
        model = TorchAlexNet(num_classes=num_classes, first_kernel=first_kernel)
    else:
        raise ValueError(f"model_name must be one of 'convolutional','fft','torch', got '{model_name}'")
    return model.to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["convolutional", "fft", "torch"],
                        help="Which models to train/benchmark: convolutional, fft, torch")
    parser.add_argument("--first-kernel", type=int, default=3, help="First (largest) kernel size")
    parser.add_argument("--epochs", type=int, default=7, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="results")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    trainloader, testloader = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)
    device = torch.device(args.device)
    criterion = nn.CrossEntropyLoss()

    results = {}

    for model_name in args.models:
        print(f"\n{'='*70}")
        print(f"Training: {model_name.upper()} | Kernel Size: {args.first_kernel} | Epochs: {args.epochs}")
        print(f"{'='*70}")
        model = build_model(model_name, args.first_kernel, num_classes=100, device=device)
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {trainable_params:,}")

        # optimizer
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        # training loop
        stats = {"train_loss": [], "train_acc": [], "val_loss": [], "val_top1": [], "val_top5": [], "time_per_batch": []}
        
        print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Top-1':>9} | {'Val Top-5':>9} | {'Batch Time':>10}")
        print(f"{'-'*5}-+-{'-'*10}-+-{'-'*9}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}")
        
        for epoch in range(args.epochs):
            train_loss, train_acc, avg_time = train_one_epoch(model, trainloader, criterion, optimizer, device)
            val_loss, val_top1, val_top5 = evaluate(model, testloader, criterion, device)
            scheduler.step()

            stats["train_loss"].append(train_loss)
            stats["train_acc"].append(train_acc)
            stats["val_loss"].append(val_loss)
            stats["val_top1"].append(val_top1)
            stats["val_top5"].append(val_top5)
            stats["time_per_batch"].append(avg_time)

            print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:8.2f}% | {val_loss:10.4f} | {val_top1:8.2f}% | {val_top5:8.2f}% | {avg_time:9.4f}s")

        # Save model with format: <topology>_<kernel_size>_<epochs>.pth
        final_path = os.path.join(args.save_dir, f"{model_name}_{args.first_kernel}_{args.epochs}.pth")
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epochs": args.epochs,
            "first_kernel": args.first_kernel,
            "final_val_top1": stats["val_top1"][-1],
            "final_val_top5": stats["val_top5"][-1]
        }, final_path)
        print(f"\n✓ Model saved to: {final_path}")

        results[model_name] = stats

    # summary
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    for m, stats in results.items():
        avg_batch_time = sum(stats["time_per_batch"]) / len(stats["time_per_batch"]) if stats["time_per_batch"] else 0.0
        final_top1 = stats["val_top1"][-1] if stats["val_top1"] else None
        final_top5 = stats["val_top5"][-1] if stats["val_top5"] else None
        print(f"{m:15s} | Avg Batch Time: {avg_batch_time:.4f}s | Final Top-1: {final_top1:.2f}% | Final Top-5: {final_top5:.2f}%")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
