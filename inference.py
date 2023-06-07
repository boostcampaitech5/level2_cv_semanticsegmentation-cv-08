import torch
import argparse
from parse_config import ConfigParser
import os
import numpy as np
from dataset import *
import pandas as pd
import torch.nn.functional as F

def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)

def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def main(config):
    path = os.path.join(config['model_dir'], config['model_file_name'])
    model = torch.load(path)
    model = model.cuda()
    model.eval()
    
    test_dataset = XRayInferenceDataset(config)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    
    CLASS2IND = {v: i for i, v in enumerate(test_dataset.CLASSES)}
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}
    
    rles = []
    filename_and_class = []
    thr = 0.5
    
    with torch.no_grad():
        n_class = len(test_dataset.CLASSES)

        for step, (images, image_names) in enumerate(test_loader):
            images = images.cuda()    
            outputs = model(images)
            
            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
    
    #to CSV
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    df.to_csv("./CSVs/output.csv", index=False)
    
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default='./config.json')
    config = ConfigParser.from_args(args)
    
    main(config)