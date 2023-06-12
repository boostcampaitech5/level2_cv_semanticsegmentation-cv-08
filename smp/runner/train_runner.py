# python native
import datetime
import os

# torch
import torch
import torch.nn.functional as F

# external library
import wandb
from tqdm.auto import tqdm

# utils
from utils.util import CLASSES, dice_coef, set_seed


def train(args, model, data_loader, val_loader, criterion, optimizer):
    print(f"Start training..")
    set_seed(args.seed)

    # Early Stop
    best_dice = 0.0
    float("inf")
    patience_limit = args.patience_limit
    patience = 0

    for epoch in range(args.epochs):
        model.train()

        # Creates a GradScaler once at the beginning of training.
        if args.fp16:
            scaler = torch.cuda.amp.GradScaler()

        for step, (images, masks) in enumerate(data_loader):
            if args.fp16:
                # Casts operations to mixed precision
                with torch.cuda.amp.autocast():
                    # gpu 연산을 위해 device 할당
                    images, masks = images.cuda(), masks.cuda()
                    model = model.cuda()

                    # inference
                    outputs = model(images)

                    # loss 계산
                    loss = criterion(outputs, masks)

                # Scales the loss, and calls backward()
                # to create scaled gradients
                scaler.scale(loss).backward()

                # Unscales gradients and calls
                # or skips optimizer.step()
                scaler.step(optimizer)

                # Updates the scale for next iteration
                scaler.update()
            else:
                # gpu 연산을 위해 device 할당
                images, masks = images.cuda(), masks.cuda()
                model = model.cuda()

                # inference
                outputs = model(images)

                # loss 계산
                loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # step 주기에 따른 loss 출력
            if (step + 1) % args.log_step == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f"Epoch [{epoch+1}/{args.epochs}], "
                    f"Step [{step+1}/{len(data_loader)}], "
                    f"Loss: {round(loss.item(),6)}"
                )
            wandb.log({'train/loss': loss.item()})

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % args.val_every == 0:
            dice = valid(args, epoch + 1, model, val_loader, criterion)

            if best_dice < dice:
                print(f"\nBest performance at epoch: {epoch + 1}, {best_dice:.6f} -> {dice:.6f}")
                print(f"Save model in {args.save_model_dir}")
                best_dice = dice
                patience = 0
                torch.save(model, os.path.join(args.save_model_dir, args.save_model_fname))
            elif dice > 0.1: # 상승하기 시작하면 count
                patience += 1
                if patience >= patience_limit: break
                            
        wandb.log({'epoch': epoch})


def valid(args, epoch, model, data_loader, criterion, thr=0.5):
    print(f"Start validation #{epoch:2d}")
    set_seed(args.seed)
    model.eval()

    dices = []
    with torch.no_grad():
        total_loss = 0
        cnt = 0

        for step, (images, masks) in enumerate(data_loader):
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()

            outputs = model(images)

            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)

            # restore original size
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach()
            masks = masks.detach()

            dice = dice_coef(outputs, masks)
            dices.append(dice)
            
            # step 주기에 따른 loss 출력
            if (step + 1) % (args.log_step//2) == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f"Epoch [{epoch+1}/{args.epochs}], "
                    f"Step [{step+1}/{len(data_loader)}], "
                    f"Loss: {round(loss.item(),6)}, ",
                    f"Dice: {round(torch.mean(dice).item(), 6)}"
                )
            wandb.log({"valid/loss": loss.item()})

    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    for idx, (c, d) in enumerate(zip(CLASSES, dices_per_class)):
        if (idx+1)%5==0: print(f"{c:<12}: {d.item():.6f}", end="\n")
        else: print(f"{c:<12}: {d.item():.6f} | ", end="")

    avg_dice = torch.mean(dices_per_class).item()

    wandb.log({"valid/dice": avg_dice})

    return avg_dice
