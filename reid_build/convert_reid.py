from pathlib import Path
from rknn.api import RKNN
ROOT=Path('/root/Develop/pose')
onnx=ROOT/'models/reid/osnet_x0_25_256x128.onnx'
out=ROOT/'models/reid/osnet_x0_25-rk3576-i8.rknn'
data=ROOT/'reid_build/dataset.txt'
rknn=RKNN(verbose=True)
print('configuring RK3576', flush=True)
ret=rknn.config(target_platform='rk3576', mean_values=[[123.675,116.28,103.53]], std_values=[[58.395,57.12,57.375]], optimization_level=3, quantized_dtype='w8a8')
assert ret==0,ret
print('loading',onnx, flush=True)
ret=rknn.load_onnx(model=str(onnx), input_size_list=[[1,3,256,128]])
assert ret==0,ret
print('building with',data, flush=True)
ret=rknn.build(do_quantization=True,dataset=str(data), rknn_batch_size=1)
assert ret==0,ret
ret=rknn.export_rknn(str(out))
assert ret==0,ret
print('exported',out,out.stat().st_size, flush=True)
rknn.release()
