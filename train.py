import argparse
import model as _model
from torch.utils.data import DataLoader
import dataset
import torch
from functools import partial
import datetime
from parse_config import ConfigParser
import torch.nn.functional as F
from metric import dice_coef
import os
import loss as module_loss
import model as CustomModel
import utils

def main(config):
    utils.set_seed(config['seed'])
    
    print(f'Start trainig.....')
    
    model = getattr(CustomModel, config['model'])(config)
    model.cuda()
    
    optimizer = partial(getattr(torch.optim, config['optimizer']['type']))
    optimizer = optimizer(model.parameters(), **config['optimizer']['parameters'])
    criterion = getattr(module_loss, config['loss'])()
    
    best_dice = 0.
    epochs = config['epochs']
    
    # train은 pickle로 가져오는게 더욱 빠르다...
    train_dataset = getattr(dataset, config['dataset'])(config, is_train=True)
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = config['train_batch_size'],
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    # validation의 경우 pickle로 가져오는게 더 오래걸린다.... 왜 이런거지??
    valid_dataset = getattr(dataset, config['dataset'])(config, is_train=False)
    valid_loader = DataLoader(
        dataset = valid_dataset,
        batch_size = config['valid_batch_size'],
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    
    for epoch in range(epochs):
        model.train()
        
        # train step
        for step, (images, masks) in enumerate(train_loader):
            image, masks = images.cuda(), masks.cuda()
            
            outputs = model.train_step(image)
            
            optimizer.zero_grad()
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            if (step+1)%config['show_train_step']==0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{epochs}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
        
        # validate step
        dices = []
        thr = 0.5
        with torch.no_grad():
            total_loss = 0
            cnt = 0
            
            model.eval()
            for step, (images, masks) in enumerate(valid_loader):
                images, masks = images.cuda(), masks.cuda()
                
                outputs = model.train_step(images)
                output_h, output_w = outputs.size(-2), outputs.size(-1)
                mask_h, mask_w = masks.size(-2), masks.size(-1)
                
                # restore original size
                if output_h != mask_h or output_w != mask_w:
                    outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
                    
                loss = criterion(outputs, masks)
                total_loss+=loss
                cnt+=1
                
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > thr).detach().cpu()
                masks = masks.detach().cpu()
                
                dice = dice_coef(outputs, masks)
                dices.append(dice)
            
            dices = torch.cat(dices, 0)
            dices_per_class = torch.mean(dices, 0)
            dice_str = [
                f"{c:<12}: {d.item():.4f}"
                for c, d in zip(valid_dataset.CLASSES, dices_per_class)
            ]
            dice_str = "\n".join(dice_str)
            print(dice_str)
            
            avg_dice = torch.mean(dices_per_class).item()
            
            if best_dice < avg_dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {avg_dice:.4f}")
                print(f"Save model in {config['model_dir']}")
                best_dice = avg_dice
                
                # save model section
                torch.save(model, os.path.join(config['model_dir'], config['model_file_name']))
                
if __name__=="__main__":    
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default='./config.json', type=str,
                    help='config file path (default: None)')
    config = ConfigParser.from_args(args)
    main(config)