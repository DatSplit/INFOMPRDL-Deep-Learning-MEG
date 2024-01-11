import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import torch
import torch.nn as nn
import math


def evaluate_model(
    model: nn.Module,
    eval_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.0
    with torch.no_grad():
        for X, y in eval_loader:
            X = X.float().to(device)
            y = y.long().to(device)

            seq_len = X.size(0)
            preds = model(X)
            total_loss += seq_len * criterion(preds, y).item()
    return total_loss / (len(eval_loader) - 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
) -> None:
    def train_epoch(model: nn.Module, epoch: int) -> None:
        model.train()  # turn on train mode
        total_loss = 0.0
        log_interval = 50
        start_time = time.time()

        num_batches = len(train_loader)
        for i_batch, (X, y) in enumerate(train_loader):
            X = X.float().to(device)
            y = y.long().to(device)

            preds = model(X)

            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            if i_batch % log_interval == 0 and i_batch > 0:
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                print(
                    f"| epoch {epoch:3d} | {i_batch:5d}/{num_batches:5d} batches | "
                    f"ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.4f}"
                )
                total_loss = 0
                start_time = time.time()

    best_val_loss = float("inf")
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        train_epoch(model, epoch)
        val_loss = evaluate_model(model, eval_loader, criterion, device)
        elapsed = time.time() - epoch_start_time
        print("-" * 89)
        print(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss {val_loss:5.2f}"
        )
        print("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/transformer_model.pth")
