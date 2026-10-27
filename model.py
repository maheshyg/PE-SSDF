import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(ci,co,3,padding=1),nn.BatchNorm2d(co),nn.ReLU(),nn.Conv2d(co,co,3,padding=1),nn.BatchNorm2d(co),nn.ReLU())
    def forward(self,x): return self.net(x)

def ellipse_prior(params, shape):
    h,w=shape
    yy,xx=torch.meshgrid(torch.linspace(0,1,h,device=params.device),torch.linspace(0,1,w,device=params.device),indexing='ij')
    cx,cy,a,b,theta=[params[:,i,None,None] for i in range(5)]
    dx=xx[None]-cx;dy=yy[None]-cy
    xr=torch.cos(theta)*dx+torch.sin(theta)*dy
    yr=-torch.sin(theta)*dx+torch.cos(theta)*dy
    return ((xr/a)**2+(yr/b)**2-1).unsqueeze(1)

class PESDFNet(nn.Module):
    """Illustrative architecture, not the original experimental implementation."""
    def __init__(self,base=32):
        super().__init__()
        ch=[base*2**i for i in range(5)]
        self.enc=nn.ModuleList([ConvBlock(1,ch[0])]+[ConvBlock(ch[i-1],ch[i]) for i in range(1,5)])
        self.param=nn.Linear(ch[-1],5)
        self.sdf=nn.Conv2d(ch[-1],1,1)
        self.fuse=ConvBlock(ch[-1]+2,ch[-1])
        self.up=nn.ModuleList([nn.ConvTranspose2d(ch[i],ch[i-1],2,2) for i in range(4,0,-1)])
        self.dec=nn.ModuleList([ConvBlock(ch[i-1]*2,ch[i-1]) for i in range(4,0,-1)])
        self.head=nn.Conv2d(ch[0],1,1)
    def forward(self,x):
        if x.shape[-1]%16 or x.shape[-2]%16: raise ValueError('Dimensions must be divisible by 16')
        features=[];h=x
        for i,layer in enumerate(self.enc):
            if i: h=F.max_pool2d(h,2)
            h=layer(h);features.append(h)
        raw=self.param(F.adaptive_avg_pool2d(h,1).flatten(1))
        params=torch.cat([torch.sigmoid(raw[:,:2]),0.05+0.45*torch.sigmoid(raw[:,2:4]),1.570796*torch.tanh(raw[:,4:5])],1)
        prior=ellipse_prior(params,h.shape[-2:]);sdf_low=self.sdf(h)
        h=self.fuse(torch.cat([h,prior,sdf_low],1))
        for i,(up,dec) in enumerate(zip(self.up,self.dec)):
            h=dec(torch.cat([up(h),features[-2-i]],1))
        return {'logits':self.head(h),'sdf':F.interpolate(sdf_low,size=x.shape[-2:],mode='bilinear',align_corners=False),'ellipse':params,'prior':ellipse_prior(params,x.shape[-2:])}
