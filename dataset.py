from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class UltrasoundDataset(Dataset):
    """Expects paired image and filled binary mask PNGs with identical filenames."""
    def __init__(self,images,masks,size=256):
        self.pairs=[];self.size=size
        for p in sorted(Path(images).glob('*.png')):
            q=Path(masks)/p.name
            if not q.exists(): raise FileNotFoundError(q)
            self.pairs.append((p,q))
        if not self.pairs: raise ValueError('No image/mask pairs found')
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        p,q=self.pairs[i]
        x=cv2.imread(str(p),0);y=cv2.imread(str(q),0)
        if x is None or y is None: raise ValueError(f'Unreadable pair {p}')
        x=cv2.medianBlur(x,3)
        x=((x.astype('float32')-x.min())/(float(x.max())-float(x.min())+1e-6)*255).astype('uint8')
        x=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(x)
        x=cv2.resize(x,(self.size,self.size),interpolation=cv2.INTER_LINEAR).astype('float32')/255
        y=(cv2.resize(y,(self.size,self.size),interpolation=cv2.INTER_NEAREST)>127).astype('float32')
        return torch.from_numpy(x[None]),torch.from_numpy(y[None])
