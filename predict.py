import argparse
import cv2
import numpy as np
import torch
from model import PESDFNet

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--image',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',default='mask.png');args=ap.parse_args()
    im=cv2.imread(args.image,0)
    if im is None: raise FileNotFoundError(args.image)
    im=cv2.medianBlur(im,3)
    im=((im.astype('float32')-im.min())/(float(im.max())-float(im.min())+1e-6)*255).astype('uint8')
    im=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(im)
    im=cv2.resize(im,(256,256)).astype('float32')/255
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=PESDFNet(ck['base']);model.load_state_dict(ck['state_dict']);model.eval()
    with torch.no_grad(): mask=(torch.sigmoid(model(torch.from_numpy(im[None,None]))['logits'])[0,0].numpy()>=0.5).astype('uint8')*255
    cv2.imwrite(args.output,mask);print('Saved',args.output)
if __name__=='__main__': main()
