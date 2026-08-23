# Phase 11 measurement pass: run every benchmark across every needed config,
# parse stdout, write one CSV per chart family. Robust: each section is wrapped so
# one missing binary doesn't kill the rest, and it prints how many rows it captured.
import subprocess, re, statistics, csv, os
def run(cmd): return subprocess.run(cmd, capture_output=True, text=True).stdout
def med(cmd, parse, reps):
    acc={}
    for _ in range(reps):
        for k,v in parse(run(cmd)).items(): acc.setdefault(k,[]).append(v)
    return {k:statistics.median(v) for k,v in acc.items() if v}
def write(fn, cols, rows):
    with open(fn,"w",newline="") as f:
        w=csv.DictWriter(f,cols); w.writeheader(); w.writerows(rows)
    print(f"  -> {fn}: {len(rows)} rows")

# M1: single-channel filter-size sweep (cnn_pipeline: naive/tiled/cuDNN vs k)
try:
    def p1(o):
        m=re.search(r"Avg Convolution time\s*:\s*([\d.]+)",o); return {"c":float(m.group(1))} if m else {}
    rows=[]
    for mode in ["custom","tiled","cudnn"]:
        for k in [3,5,7,9,11,15,19,23,27,31]:
            d=med(["./cnn_pipeline",f"--mode={mode}",f"--ksize={k}"],p1,7)
            if "c" in d: rows.append({"ksize":k,"mode":mode,"conv_ms":round(d["c"],5)})
    print("M1 single-channel filter sweep"); write("data_sweep_sc.csv",["ksize","mode","conv_ms"],rows)
except Exception as ex: print("M1 FAILED:",ex)

# M2: single-channel per-layer (custom vs cuDNN FP32/FP16)
try:
    def p2a(o):
        d={}
        for L in ["Convolution","BatchNorm","ReLU","Pooling"]:
            m=re.search(rf"Avg {L} time\s*:\s*([\d.]+)",o)
            if m: d[L]=float(m.group(1))
        return d
    def p2b(o):
        return {f"{m.group(1)}_{m.group(2)}":float(m.group(3)) for m in re.finditer(r"(FP32|FP16)\s+(\w+)\s+avg:\s*([\d.]+)",o)}
    cus=med(["./cnn_pipeline","--mode=custom","--ksize=3"],p2a,5)
    cud=med(["./cudnn_pipeline"],p2b,5)
    lm={"Convolution":"Conv","BatchNorm":"BN","ReLU":"ReLU","Pooling":"Pool"}
    rows=[]
    for L,sh in lm.items():
        if L in cus: rows.append({"layer":sh,"impl":"custom","ms":round(cus[L],5)})
        if f"FP32_{sh}" in cud: rows.append({"layer":sh,"impl":"cudnn_fp32","ms":round(cud[f"FP32_{sh}"],5)})
        if f"FP16_{sh}" in cud: rows.append({"layer":sh,"impl":"cudnn_fp16","ms":round(cud[f"FP16_{sh}"],5)})
    print("M2 single-channel per-layer"); write("data_perlayer_sc.csv",["layer","impl","ms"],rows)
except Exception as ex: print("M2 FAILED:",ex)

# M3: multi-channel conv sweep (naive/tiled/coarse/im2col/cuDNN/cuDNN16 vs C)
try:
    def p3(o):
        d={}
        for ln in o.splitlines():
            m=re.match(r"\s*([A-Za-z0-9]+)\s*:\s*([\d.]+)\s*ms",ln)
            if m: d[m.group(1)]=float(m.group(2))
        return d
    rows=[]
    for C in [1,8,16,32,64,128]:
        d=med(["./mc_conv_bench",f"--cin={C}",f"--cout={C}","--hw=256"],p3,3)
        for m in ["naive","tiled","coarse","im2col","cuDNN","cuDNN16"]:
            if m in d: rows.append({"C":C,"method":m,"conv_ms":round(d[m],5)})
    print("M3 multi-channel conv sweep"); write("data_mc_conv.csv",["C","method","conv_ms"],rows)
except Exception as ex: print("M3 FAILED:",ex)

# M4: fusion (epilogue + full, unfused vs fused)
try:
    def p4(o):
        d={}
        for k in ["epilogue","full"]:
            m=re.search(rf"{k}\s+unfused\s+([\d.]+) ms\s+fused\s+([\d.]+) ms",o)
            if m: d[f"{k}_unfused"]=float(m.group(1)); d[f"{k}_fused"]=float(m.group(2))
        return d
    rows=[]
    for C in [16,32,64,128]:
        d=med(["./fuse_bench",f"--cin={C}",f"--cout={C}","--hw=256"],p4,3)
        for k,v in d.items(): rows.append({"C":C,"metric":k,"ms":round(v,5)})
    print("M4 fusion"); write("data_fusion.csv",["C","metric","ms"],rows)
except Exception as ex: print("M4 FAILED:",ex)

# M5: coarsening-factor sweep
try:
    def p5(o):
        return {int(m.group(1)):float(m.group(2)) for m in re.finditer(r"COARSEN=(\d+)\s*:\s*([\d.]+) ms",o)}
    rows=[]
    for C in [64,128]:
        d=med(["./coarsen_sweep",f"--c={C}","--hw=256"],p5,3)
        for cf,v in sorted(d.items()): rows.append({"C":C,"coarsen":cf,"conv_ms":round(v,5)})
    print("M5 coarsening sweep"); write("data_coarsen_sweep.csv",["C","coarsen","conv_ms"],rows)
except Exception as ex: print("M5 FAILED:",ex)

# M6: TensorRT (PyTorch vs TRT FP32/TF32 vs TRT FP16) -- slow (engine builds)
try:
    def p6(o):
        d={}
        for lab,key in [("PyTorch eager","pytorch"),("TRT FP32/TF32","trt_fp32"),("TRT FP16","trt_fp16")]:
            m=re.search(re.escape(lab)+r"\s*:\s*([\d.]+)",o)
            if m: d[key]=float(m.group(1))
        return d
    d=p6(run(["python","trt_pipeline.py"]))
    rows=[{"impl":k,"ms":round(v,5)} for k,v in d.items()]
    print("M6 TensorRT"); write("data_trt.csv",["impl","ms"],rows)
except Exception as ex: print("M6 FAILED:",ex)

print("\nSummary:")
for fn in ["data_sweep_sc.csv","data_perlayer_sc.csv","data_mc_conv.csv","data_fusion.csv","data_coarsen_sweep.csv","data_trt.csv"]:
    print(f"  {fn}: {sum(1 for _ in open(fn))-1 if os.path.exists(fn) else 'MISSING'} rows")
