# python native
import datetime
import os
import sys
import time

# torch
import torch
import torch.nn.functional as F

# external library
import wandb
from tqdm.auto import tqdm

sys.path.append("..")
from loss import dice_coef

# utils
from utils import CLASSES, set_seed


def train(config, model, data_loader, val_loader, criterion, optimizer):
    model_name = config.smp.model if config.smp.use_smp else config.model
    print(
        f"Start training..\n"
        f"model : {model_name if not config.resume_from else config.resume_from}\n"
        f"epochs : {config.epochs}\n"
        f"batch size : {config.train_batch_size}\n"
        f"fp16 : {config.fp16}\n"
        f"Gradient Accumulation Step : {config.accumulation_step}\n",
    )
    set_seed(config.seed)

    # Early Stop
    best_dice = 0.0
    patience_limit = config.patience_limit
    patience = 0

    if config.fp16:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.epochs):
        st = time.time()
        if config.wandb.use_wandb:
            wandb.log({"epoch": epoch})
        model.train()

        for step, (images, masks) in enumerate(data_loader):
            if config.fp16:
                with torch.cuda.amp.autocast():
                    images, masks = images.cuda(), masks.cuda()
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                if ((step + 1) % config.accumulation_step == 0) or ((step + 1) == len(data_loader)):
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scaler.update()
            else:
                images, masks = images.cuda(), masks.cuda()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                if ((step + 1) % config.accumulation_step == 0) or ((step + 1) == len(data_loader)):
                    optimizer.step()
                    optimizer.zero_grad()

            if (step + 1) % config.log_step == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f"Epoch [{epoch+1}/{config.epochs}], "
                    f"Step [{step+1}/{len(data_loader)}], "
                    f"Loss: {round(loss.item(),6)}"
                )
                if config.wandb.use_wandb:
                    wandb.log({"train/loss": loss.item()})
        if (epoch + 1) % config.val_every == 0:
            dice = valid(config, epoch + 1, model, val_loader, criterion)

            if best_dice <= dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.6f} -> {dice:.6f}")
                print(f"Save model in {config.model_dir}")
                best_dice = dice
                patience = 0
                torch.save(model, os.path.join(config.model_dir, "best.pt"))
            else:
                patience += 1
                if patience >= patience_limit:
                    print(f"over {patience_limit}, Early Stopping....")
                    break
        ed = time.time()
        torch.save(model, os.path.join(config.model_dir, "last.pt"))
        print(f"Epoch {epoch} : {(ed-st)} s")


def valid(config, epoch, model, data_loader, criterion, thr=0.5):
    print(f"Start validation #{epoch:2d}")
    set_seed(config.seed)
    model.eval()

    dices = []
    with torch.no_grad():
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()

            outputs = model(images)

            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)

            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()

            dice = dice_coef(outputs, masks)
            dices.append(dice)

            if config.wandb.use_wandb:
                wandb.log({"valid/loss": loss.item()})

    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [f"{c:<12}: {d.item():.6f}" for c, d in zip(CLASSES, dices_per_class)]
    dice_str = "\n".join(dice_str)
    print(dice_str)

    avg_dice = torch.mean(dices_per_class).item()

    if config.wandb.use_wandb:
        wandb.log({"valid/dice": avg_dice})

    return avg_dice
