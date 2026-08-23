# TensorRT sweep: PyTorch -> ONNX -> TRT (TF32 + FP16) across channel counts.
# Writes data_trt.csv with columns C,impl,ms. TRT 11 is strongly-typed: precision
# comes from the ONNX (FP32 ONNX -> TF32 engine; FP16 ONNX -> FP16 engine).
import torch, torch.nn as nn, tensorrt as trt, csv, re
dev='cuda'; torch.manual_seed(0)
H,W,N,R,S = 256,256,1,3,3

class CNN(nn.Module):
    def __init__(s,cin,cout):
        super().__init__()
        s.conv=nn.Conv2d(cin,cout,3,padding=0,bias=False); s.bn=nn.BatchNorm2d(cout)
        s.relu=nn.ReLU(); s.pool=nn.MaxPool2d(2,2)
    def forward(s,x): return s.pool(s.relu(s.bn(s.conv(x))))

def bench(fn,warm=10,it=50):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(it): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b)/it

LOG=trt.Logger(trt.Logger.ERROR)
def make_net(bld):
    F=trt.NetworkDefinitionCreationFlag
    for fl in ("STRONGLY_TYPED","EXPLICIT_BATCH"):
        if hasattr(F,fl): return bld.create_network(1<<int(getattr(F,fl)))
    return bld.create_network(0)
def build(onnx):
    bld=trt.Builder(LOG); net=make_net(bld); p=trt.OnnxParser(net,LOG)
    with open(onnx,"rb") as f:
        if not p.parse(f.read()): raise RuntimeError("parse")
    cfg=bld.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<30)
    return trt.Runtime(LOG).deserialize_cuda_engine(bld.build_serialized_network(net,cfg))
_T2T={trt.DataType.FLOAT:torch.float32,trt.DataType.HALF:torch.float16}
def runner(eng,inp):
    ctx=eng.create_execution_context(); out=None
    for i in range(eng.num_io_tensors):
        nm=eng.get_tensor_name(i)
        if eng.get_tensor_mode(nm)==trt.TensorIOMode.INPUT: ctx.set_tensor_address(nm,inp.data_ptr())
        else:
            sh=tuple(eng.get_tensor_shape(nm))
            out=torch.empty(sh,dtype=_T2T.get(eng.get_tensor_dtype(nm),torch.float32),device=dev)
            ctx.set_tensor_address(nm,out.data_ptr())
    st=torch.cuda.current_stream().cuda_stream
    return (lambda: ctx.execute_async_v3(stream_handle=st)), out

rows=[]
for C in [16,32,64,128]:
    m=CNN(C,C).to(dev).eval()
    with torch.no_grad():
        m.bn.running_mean.normal_(); m.bn.running_var.uniform_(0.5,1.5)
        m.bn.weight.uniform_(0.5,1.5); m.bn.bias.normal_()
    x=torch.randn(N,C,H,W,device=dev)
    with torch.no_grad(): t_torch=bench(lambda: m(x))
    torch.onnx.export(m,x,"m.onnx",input_names=["i"],output_names=["o"],opset_version=17,dynamo=False)
    mh=CNN(C,C).to(dev).eval(); mh.load_state_dict(m.state_dict()); mh.half()
    torch.onnx.export(mh,x.half(),"m16.onnx",input_names=["i"],output_names=["o"],opset_version=17,dynamo=False)
    r32,_=runner(build("m.onnx"),x);      t32=bench(r32)
    r16,_=runner(build("m16.onnx"),x.half()); t16=bench(r16)
    print(f"C={C:4d}  PyTorch {t_torch:.4f}  TF32 {t32:.4f}  FP16 {t16:.4f} ms")
    rows += [{"C":C,"impl":"pytorch","ms":round(t_torch,5)},
             {"C":C,"impl":"trt_fp32","ms":round(t32,5)},
             {"C":C,"impl":"trt_fp16","ms":round(t16,5)}]

with open("data_trt.csv","w",newline="") as f:
    w=csv.DictWriter(f,["C","impl","ms"]); w.writeheader(); w.writerows(rows)
print(f"\ndata_trt.csv: {len(rows)} rows")
