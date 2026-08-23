# TensorRT end-to-end pipeline: Conv -> BN -> ReLU -> MaxPool
#   Production workflow: PyTorch -> ONNX -> TensorRT engine.
#   TensorRT 11 removed weak-typing precision flags (BuilderFlag.FP16 etc.): every
#   network is strongly typed, so precision comes from the ONNX. FP32 ONNX -> a
#   TF32/FP32 engine; FP16 ONNX -> an FP16 engine. Output validated vs PyTorch.
import torch, torch.nn as nn
import tensorrt as trt

dev='cuda'; torch.manual_seed(0)
Cin,Cout,H,W,N = 64,64,256,256,1

class CNN(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.conv=nn.Conv2d(cin,cout,3,padding=0,bias=False)   # valid conv (matches our kernels)
        self.bn=nn.BatchNorm2d(cout); self.relu=nn.ReLU(); self.pool=nn.MaxPool2d(2,2)
    def forward(self,x): return self.pool(self.relu(self.bn(self.conv(x))))

model=CNN(Cin,Cout).to(dev).eval()
with torch.no_grad():                          # non-trivial BN so it isn't an optimizable no-op
    model.bn.running_mean.normal_(); model.bn.running_var.uniform_(0.5,1.5)
    model.bn.weight.uniform_(0.5,1.5); model.bn.bias.normal_()
x=torch.randn(N,Cin,H,W,device=dev)

def bench(fn,warmup=10,iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters

with torch.no_grad():
    ref=model(x); t_torch=bench(lambda: model(x))
print(f"Config: N={N} Cin={Cin} Cout={Cout} {H}x{W}  output {tuple(ref.shape)}")
print(f"  PyTorch eager : {t_torch:.4f} ms")

torch.onnx.export(model,x,"model.onnx",input_names=["input"],output_names=["output"],opset_version=17,dynamo=False)
model_h=CNN(Cin,Cout).to(dev).eval(); model_h.load_state_dict(model.state_dict()); model_h.half()
torch.onnx.export(model_h,x.half(),"model_fp16.onnx",input_names=["input"],output_names=["output"],opset_version=17,dynamo=False)
print("  exported FP32 + FP16 ONNX")

TRT_LOGGER=trt.Logger(trt.Logger.WARNING)
def make_network(b):
    F=trt.NetworkDefinitionCreationFlag
    for flag in ("STRONGLY_TYPED","EXPLICIT_BATCH"):
        if hasattr(F,flag): return b.create_network(1<<int(getattr(F,flag)))
    return b.create_network(0)

def build_engine(onnx_path):
    b=trt.Builder(TRT_LOGGER); net=make_network(b)
    p=trt.OnnxParser(net,TRT_LOGGER)
    with open(onnx_path,"rb") as f:
        if not p.parse(f.read()):
            for i in range(p.num_errors): print("   ONNX parse error:",p.get_error(i))
            raise RuntimeError("parse failed")
    cfg=b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<30)
    ser=b.build_serialized_network(net,cfg)
    if ser is None: raise RuntimeError("build failed")
    return trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(ser)

_T2T={trt.DataType.FLOAT:torch.float32, trt.DataType.HALF:torch.float16}
def make_runner(engine,inp):
    ctx=engine.create_execution_context(); out=None
    for i in range(engine.num_io_tensors):
        name=engine.get_tensor_name(i)
        if engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT:
            ctx.set_tensor_address(name,inp.data_ptr())
        else:
            shape=tuple(engine.get_tensor_shape(name))
            out=torch.empty(shape,dtype=_T2T.get(engine.get_tensor_dtype(name),torch.float32),device=dev)
            ctx.set_tensor_address(name,out.data_ptr())
    st=torch.cuda.current_stream().cuda_stream
    return (lambda: ctx.execute_async_v3(stream_handle=st)), out

for tag,onnx_path,inp in [("TRT FP32/TF32","model.onnx",x), ("TRT FP16","model_fp16.onnx",x.half())]:
    eng=build_engine(onnx_path)
    run,out=make_runner(eng,inp); run(); torch.cuda.synchronize()
    a,bb=out.float().flatten(),ref.float().flatten()
    rel=(torch.norm(a-bb)/(torch.norm(bb)+1e-12)).item()
    t=bench(run)
    print(f"  {tag}: {t:.4f} ms   rel_L2={rel:.3e}   {t_torch/t:.2f}x vs PyTorch   {'PASS' if rel<2e-2 else 'MISMATCH'}")
