import sys,torch
from pathlib import Path
sys.path.insert(0,'/root/Develop/pose/reid_build')
from osnet import osnet_x0_25
weights=Path('/root/Develop/pose/reid_build/osnet_x0_25_imagenet.pth')
out=Path('/root/Develop/pose/models/reid/osnet_x0_25_256x128.onnx')
ckpt=torch.load(weights,map_location='cpu')
model=osnet_x0_25(num_classes=1000,pretrained=False)
if isinstance(ckpt,dict) and 'state_dict' in ckpt: ckpt=ckpt['state_dict']
if isinstance(ckpt,dict) and 'model' in ckpt and isinstance(ckpt['model'],dict): ckpt=ckpt['model']
ckpt={k.replace('module.','',1):v for k,v in ckpt.items()}
missing,unexpected=model.load_state_dict(ckpt,strict=False)
print('missing',missing,'unexpected',unexpected)
class Embedding(torch.nn.Module):
 def __init__(self,m): super().__init__(); self.m=m
 def forward(self,x): return self.m(x)
wrapped=Embedding(model.eval())
x=torch.randn(1,3,256,128)
with torch.no_grad(): print('embedding',wrapped(x).shape)
torch.onnx.export(wrapped,x,str(out),input_names=['images'],output_names=['embedding'],opset_version=12,dynamic_axes=None)
print(out,out.stat().st_size)
