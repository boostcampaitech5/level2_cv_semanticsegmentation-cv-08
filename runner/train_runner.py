# python native
import collections
import datetime
import os
import time

# torch
import torch
import torch.nn.functional as F

# external library
import wandb

# utils
from loss.metric import dice_coef
from utils import CLASSES, set_seed


def train(config, model, data_loader, val_loader, criterion, optimizer, lr_scheduler):
    model_name = config.base.smp.model if config.base.use == "smp" else config.base.pytorch.model
    print(
        f"Start training ...\n"
        f"model\t: {model_name if not config.resume_from else config.resume_from}\n"
        f"epochs\t: {config.epochs}\n"
        f"batch size : {config.train.batch_size}\n"
        f"criterion : {config.criterion}\n"
        f"scheduler : {config.scheduler.type}\n"
        f"fp16\t: {config.fp16}\n"
        f"Gradient Accumulation Step : {config.accumulation_step}\n",
    )
    set_seed(config.seed)

    # Early Stop
    best_dice = 0.0
    best_epoch = 0
    patience_limit = config.patience_limit
    patience = 0

    if config.fp16:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.epochs):
        st = time.time()

        if config.wandb.use:
            wandb.log({"epoch": epoch})

        model.train()

        for step, (images, masks) in enumerate(data_loader):
            if config.fp16:
                with torch.cuda.amp.autocast():
                    # gpu 연산을 위해 device 할당
                    images, masks = images.cuda(), masks.cuda()
                    model = model.cuda()

                    # inference
                    outputs = model(images)

                    if isinstance(outputs, collections.OrderedDict):
                        outputs = outputs["out"]

                    if (config.base.use == "pytorch" and (
                        config.base.pytorch.model.startswith("hrnet")
                        or config.base.pytorch.model.startswith("Mask"))) or (
                            config.base.use == "smp" and config.base.smp.parameters.encoder_name == "swin_encoder"
                        ):
                        output_h, output_w = outputs.size(-2), outputs.size(-1)
                        mask_h, mask_w = masks.size(-2), masks.size(-1)

                        # restore original size
                        if output_h != mask_h or output_w != mask_w:
                            outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

                    # loss 계산
                    if config.base.use == "pytorch" and config.base.pytorch.model in [
                        "UNet_3Plus_DeepSup",
                        "UNet_3Plus_DeepSup_CGM",
                    ]:
                        loss_list = [criterion(output, masks) for output in outputs]
                        loss = torch.mean(torch.stack(loss_list))
                    else:
                        loss = criterion(outputs, masks)

                scaler.scale(loss).backward()

                if ((step + 1) % config.accumulation_step == 0) or ((step + 1) == len(data_loader)):
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scaler.update()
            else:
                # gpu 연산을 위해 device 할당
                images, masks = images.cuda(), masks.cuda()
                model = model.cuda()

                # inference
                outputs = model(images)

                if isinstance(outputs, collections.OrderedDict):
                    outputs = outputs["out"]

                if config.base.use == "pytorch" and (
                    config.base.pytorch.model.startswith("hrnet")
                    or config.base.pytorch.model.startswith("Mask")
                ):
                    output_h, output_w = outputs.size(-2), outputs.size(-1)
                    mask_h, mask_w = masks.size(-2), masks.size(-1)

                    # restore original size
                    if output_h != mask_h or output_w != mask_w:
                        outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

                # loss 계산
                if config.base.use == "pytorch" and config.base.pytorch.model in [
                    "UNet_3Plus_DeepSup",
                    "UNet_3Plus_DeepSup_CGM",
                ]:
                    loss_list = [criterion(output, masks) for output in outputs]
                    loss = torch.mean(torch.stack(loss_list))
                else:
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

            if config.wandb.use:
                wandb.log({"train/loss": loss.item()})

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % config.val_every == 0:
            dice = valid(config, epoch + 1, model, val_loader, criterion)

            if config.scheduler.type == "ReduceLROnPlateau":
                lr_scheduler.step(dice)
            else:
                lr_scheduler.step()

            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.6f} -> {dice:.6f}")
                best_dice = dice
                best_epoch = epoch + 1
                patience = 0

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": lr_scheduler.state_dict(),
                    },
                    os.path.join(config.save_model_dir, config.model_file_name),
                )
            elif dice > 0.1:  # 상승하기 시작하면 count
                patience += 1
                if patience >= patience_limit:
                    print(f"Over {patience_limit}, Early Stopping ...")
                    break

        ed = time.time()

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
            },
            os.path.join(config.save_model_dir, "last.pth"),
        )
        print(f"Epoch {epoch+1} : {(ed-st)} s")

    print(f"Done !!")
    print(f"Best performance at epoch: {best_epoch} dice: {best_dice:.6f}")
    print(f"Save model in {config.save_model_dir}")


def valid(config, epoch, model, data_loader, criterion, thr=0.5):
    print(f"Start validation #{epoch:2d}")
    set_seed(config.seed)
    model.eval()

    dices = []
    with torch.no_grad():
        total_loss = 0
        cnt = 0

        for step, (images, masks) in enumerate(data_loader):
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()

            outputs = model(images)
            if isinstance(outputs, collections.OrderedDict):
                outputs = outputs["out"]
            if isinstance(outputs, tuple):  # UNet 3+ Deep supervision
                outputs = outputs[0]

            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)

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
            if (step + 1) % (config.log_step // 2) == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f"Epoch [{epoch}/{config.epochs}], "
                    f"Step [{step+1}/{len(data_loader)}], "
                    f"Loss: {round(loss.item(),6)}, ",
                    f"Dice: {round(torch.mean(dice).item(), 6)}",
                )

            if config.wandb.use:
                wandb.log({"valid/loss": loss.item()})

    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    if config.wandb.use:
        wandb.log({c: d.item() for c, d in zip(CLASSES, dices_per_class)})

    for idx, (c, d) in enumerate(zip(CLASSES, dices_per_class)):
        if (idx + 1) % 5 == 0:
            print(f"{c:<12}: {d.item():.6f}", end="\n")
        else:
            print(f"{c:<12}: {d.item():.6f} | ", end="")
    print()

    avg_dice = torch.mean(dices_per_class).item()

    if config.wandb.use:
        wandb.log({"valid/dice": avg_dice})

    return avg_dice
