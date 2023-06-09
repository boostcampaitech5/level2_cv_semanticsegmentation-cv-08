# external library
from tqdm.auto import tqdm

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")
# utils
from utils import CLASSES, IND2CLASS, set_seed, encode_mask_to_rle

def test(config, model, data_loader, thr=0.5):
    set_seed(config.seed)
    model = model.cuda()
    model.eval()
        
    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
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
                    
    return rles, filename_and_class