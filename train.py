import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import PESDFNet
from dataset import UltrasoundDataset


def loss_fn(out,mask):
    prob=torch.sigmoid(out['logits'])
    dice=1-(2*(prob*mask).sum((1,2,3))+1)/((prob+mask).sum((1,2,3))+1)
    ellipse=F.binary_cross_entropy_with_logits(-out['prior'],mask)
    dx=out['sdf'][:,:,:,1:]-out['sdf'][:,:,:,:-1]
    dy=out['sdf'][:,:,1:,:]-out['sdf'][:,:,:-1,:]
    smooth=dx.abs().mean()+dy.abs().mean()
    return F.binary_cross_entropy_with_logits(out['logits'],mask)+dice.mean()+0.1*ellipse+0.01*smooth

def main():
    ap=argparse.ArgumentParser()
    for k in ['train-images','train-masks','val-images','val-masks']: ap.add_argument('--'+k,required=True)
    ap.add_argument('--epochs',type=int,default=200);ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--base',type=int,default=32);ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--output',default='best_model.pt');args=ap.parse_args()
    torch.manual_seed(42);np.random.seed(42)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=PESDFNet(args.base).to(device);opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    train=DataLoader(UltrasoundDataset(args.train_images,args.train_masks),batch_size=args.batch_size,shuffle=True)
    val=DataLoader(UltrasoundDataset(args.val_images,args.val_masks),batch_size=args.batch_size)
    best=float('inf')
    for epoch in range(args.epochs):
        model.train();tr=[]
        for x,y in train:
            x,y=x.to(device),y.to(device);opt.zero_grad();loss=loss_fn(model(x),y);loss.backward();opt.step();tr.append(loss.item())
        model.eval();vl=[]
        with torch.no_grad():
            for x,y in val: vl.append(loss_fn(model(x.to(device)),y.to(device)).item())
        score=float(np.mean(vl));print(f'Epoch {epoch+1}: train={np.mean(tr):.4f} val={score:.4f}',flush=True)
        if score<best:
            best=score;torch.save({'state_dict':model.state_dict(),'base':args.base},args.output)
if __name__=='__main__': main()
