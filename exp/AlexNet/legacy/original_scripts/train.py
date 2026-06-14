import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset

from alexnet import AlexNet


def pil_loader(path: str) -> Image.Image:
    """安全加载图像，损坏则返回 None"""
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        return img
    except (UnidentifiedImageError, OSError):
        return None


class SafeImageFolder(VisionDataset):
    """带容错的 ImageFolder，自动跳过损坏图像"""

    def __init__(self, root: str, transform=None):
        super().__init__(root, transform=transform)
        self.samples = []
        self.classes = sorted(os.listdir(root))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = pil_loader(path)
        if img is None:
            return None  # 损坏图像，由 collate_fn 过滤
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.samples)


def safe_collate(batch):
    """过滤掉 None（损坏图像）"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)


def get_dataloaders(data_dir: str, batch_size: int, num_workers: int = 2):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = SafeImageFolder(train_dir, transform=train_transform)
    val_dataset = SafeImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              collate_fn=safe_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            collate_fn=safe_collate)

    return train_loader, val_loader, len(train_dataset), len(val_dataset)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        if batch is None:
            continue
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        if batch is None:
            continue
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def plot_curves(train_losses, train_accs, val_losses, val_accs, save_path: str):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"训练曲线已保存至: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="AlexNet 猫狗二分类训练")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="数据集根目录 (包含 train/ 和 val/ 子目录)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="数据加载线程数")
    parser.add_argument("--save_dir", type=str, default=".",
                        help="模型和曲线保存目录")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_loader, val_loader, train_size, val_size = get_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    print(f"训练集: {train_size} 张, 验证集: {val_size} 张")

    model = AlexNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    best_model_path = os.path.join(args.save_dir, "best_alexnet.pth")
    curve_path = os.path.join(args.save_dir, "training_curves.png")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> 保存最佳模型 (val_acc={best_val_acc:.4f})")

    print(f"\n训练完成，最佳验证准确率: {best_val_acc:.4f}")
    plot_curves(train_losses, train_accs, val_losses, val_accs, curve_path)


if __name__ == "__main__":
    main()
