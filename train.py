#!/usr/bin/env python3
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, StepLR, SequentialLR
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from src.data.dataset import BrainTumor
from src.utils.augmentations import get_grey_transforms, get_val_transforms
from src.models.cnn import CNN

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss/total, correct/total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss/total, correct/total

def main(args=None):
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',     type=str,   default='data/kaggle')
    p.add_argument('--output_dir',   type=str,   default='checkpoints')
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--learning_rate',type=float, default=1e-3)
    p.add_argument('--reg',          type=float, default=1e-4,
                   help="weight decay (L2 regularization)")
    p.add_argument('--epochs',       type=int,   default=20)
    p.add_argument('--steps',        type=int,   default=10,
                   help="StepLR interval (in epochs)")
    p.add_argument('--warmup',       type=int,   default=5,
                   help="number of warmup epochs")
    p.add_argument('--momentum',     type=float, default=0.9)
    if args is None:
        args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- data
    train_ds = BrainTumor(args.data_dir, split='train', transform=get_grey_transforms())
    val_ds   = BrainTumor(args.data_dir, split='test',  transform=get_val_transforms())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    # -- model + loss
    model     = CNN(num_classes=4).to(device)
    criterion = CrossEntropyLoss()

    # -- optimizer
    optimizer = SGD(model.parameters(),
                    lr=args.learning_rate,
                    momentum=args.momentum,
                    weight_decay=args.reg)

    # -- scheduler: linear warmup, then step decay
    schedulers = []
    lr_lamb = None
    if args.warmup > 0:
        warmup_sched = LinearLR(optimizer,
                                start_factor=0.001,
                                end_factor=1.0,
                                total_iters=args.warmup)
        schedulers.append(warmup_sched)
    decay_sched = StepLR(optimizer,
                        step_size=args.steps,
                        gamma=0.1)
    schedulers.append(decay_sched)

    if len(schedulers) == 2:
        scheduler = SequentialLR(optimizer,
                                 schedulers=schedulers,
                                 milestones=[args.warmup])
    else:
        scheduler = schedulers[0]

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc = 0.0

    # -- training loop
    train_losses, train_accs = [], []
    val_losses,   val_accs   = [], []

    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, val_loader,   criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"Train loss={train_loss:.4f}, acc={train_acc:.4f}  "
              f" Val loss={val_loss:.4f}, acc={val_acc:.4f}")

        # checkpoint best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(model.state_dict(), ckpt)
            print(f"→ Saved best model to {ckpt}")

    return train_losses, train_accs, val_losses, val_accs

if __name__ == '__main__':
    main()
