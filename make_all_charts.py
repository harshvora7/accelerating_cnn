"""
Generate all 12 portfolio charts for accelerating_cnn from the Phase-11 CSVs.
Run in the repo root after measure_all.py. Charts are written to charts/.
Nsight-measured constants and the L4 hardware peaks are set below -- edit if needed.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, os

OUT = "charts"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.titlesize":13,
    "axes.titleweight":"bold","axes.edgecolor":"#444","figure.dpi":150,"savefig.dpi":150,
    "savefig.bbox":"tight"})

CLR = {"naive":"#E4572E","custom":"#E4572E","tiled":"#3AA655","coarse":"#8C6BAE",
       "im2col":"#E8A400","cuDNN":"#4C78A8","cudnn_fp32":"#4C78A8","cuDNN16":"#B279A7",
       "cudnn_fp16":"#B279A7","pytorch":"#999999","trt_fp32":"#4C78A8","trt_fp16":"#B279A7"}
LBL = {"naive":"Naive","custom":"Custom","tiled":"Tiled","coarse":"Coarsened","im2col":"im2col+cuBLAS",
       "cuDNN":"cuDNN","cudnn_fp32":"cuDNN FP32","cuDNN16":"cuDNN FP16","cudnn_fp16":"cuDNN FP16",
       "pytorch":"PyTorch eager","trt_fp32":"TensorRT TF32","trt_fp16":"TensorRT FP16"}

# ---- L4 datasheet peaks (edit if needed) ----
L4_FP32_TFLOPS = 30.3
L4_BW_GBs      = 300.0
# ---- Nsight-measured (single-channel, 11x11) ----
SC_L1_MB   = {"naive":205.13, "tiled":4.71}
SC_DRAM_MB = {"naive":1.54,  "tiled":1.41}
# ---- multi-channel DRAM bytes @ C=64 (MB): naive measured; add others from ncu for a fully measured roofline
MC_DRAM_MB = {"naive":25.63, "tiled":26.80, "coarse":28.68, "im2col":167.05}
ROOF_C = 64

def load(fn): return pd.read_csv(fn) if os.path.exists(fn) else None
sweep=load("data_sweep_sc.csv"); perlay=load("data_perlayer_sc.csv"); mc=load("data_mc_conv.csv")
fusion=load("data_fusion.csv"); coarsen=load("data_coarsen_sweep.csv"); trt=load("data_trt.csv")

def despine(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3, linewidth=0.7); ax.set_axisbelow(True)
def title(ax, main, sub=None):
    ax.text(0.5,1.11 if sub else 1.03, main, transform=ax.transAxes, ha="center", fontsize=13, fontweight="bold")
    if sub: ax.text(0.5,1.03, sub, transform=ax.transAxes, ha="center", fontsize=9, color="#666")
def save(fig,name): fig.savefig(f"{OUT}/{name}.png"); plt.close(fig); print("  wrote",name)
def conv_flops(C,H=256,R=3,S=3):
    Ho,Wo=H-R+1,H-S+1; return 2.0*C*Ho*Wo*C*R*S

made=0
if sweep is not None:
    fig,ax=plt.subplots(figsize=(8.2,5.2)); despine(ax)
    p=sweep.pivot(index="ksize",columns="mode",values="conv_ms")
    for m,key in [("custom","naive"),("tiled","tiled"),("cudnn","cuDNN")]:
        if m in p.columns: ax.plot(p.index,p[m],"-o",color=CLR[key],lw=2.4,ms=6,label=LBL[key])
    ax.set_yscale("log"); ax.set_xticks(p.index)
    ax.set_xlabel("Convolution filter size (k × k)"); ax.set_ylabel("Conv time (ms, log) — lower better")
    title(ax,"Single channel: the tiling / cuDNN crossover","512×512 · NVIDIA L4")
    ax.legend(frameon=False); save(fig,"01_sc_crossover"); made+=1

if perlay is not None:
    fig,ax=plt.subplots(figsize=(8.6,5.2)); despine(ax)
    p=perlay.pivot(index="layer",columns="impl",values="ms")
    order=[l for l in ["Conv","BN","ReLU","Pool"] if l in p.index]; p=p.loc[order]
    impls=[i for i in ["custom","cudnn_fp32","cudnn_fp16"] if i in p.columns]
    x=np.arange(len(order)); w=0.8/len(impls)
    for j,imp in enumerate(impls):
        ax.bar(x+(j-(len(impls)-1)/2)*w, p[imp], w, color=CLR[imp], label=LBL[imp])
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylabel("Time (ms, log) — lower better")
    title(ax,"Single channel, per layer: custom vs cuDNN","C=1 — too small to exercise cuDNN / Tensor Cores")
    ax.legend(frameon=False,ncol=3); save(fig,"02_sc_perlayer"); made+=1

if mc is not None:
    fig,ax=plt.subplots(figsize=(8.4,5.4)); despine(ax)
    p=mc.pivot(index="C",columns="method",values="conv_ms")
    for m in ["naive","tiled","coarse","im2col","cuDNN","cuDNN16"]:
        if m in p.columns: ax.plot(p.index,p[m],"-o",color=CLR[m],lw=2.2,ms=6,label=LBL[m])
    ax.set_xscale("log",base=2); ax.set_yscale("log"); ax.set_xticks(p.index); ax.set_xticklabels(p.index)
    ax.set_xlabel("Channels (Cin = Cout)"); ax.set_ylabel("Conv time (ms, log) — lower better")
    title(ax,"Multi-channel conv: every implementation vs channel count","256×256 · 3×3 · NVIDIA L4")
    ax.legend(frameon=False,ncol=2,fontsize=9); save(fig,"03_mc_ladder"); made+=1

frames=[]
if mc is not None:
    pm=mc.pivot(index="C",columns="method",values="conv_ms")
    if "cuDNN" in pm and "cuDNN16" in pm: frames.append(("cuDNN conv",(pm["cuDNN"]/pm["cuDNN16"]),"#4C78A8"))
if trt is not None:
    pt=trt.pivot(index="C",columns="impl",values="ms")
    if "trt_fp32" in pt and "trt_fp16" in pt: frames.append(("TensorRT engine",(pt["trt_fp32"]/pt["trt_fp16"]),"#111111"))
if frames:
    fig,ax=plt.subplots(figsize=(8,5)); despine(ax)
    for lab,ser,c in frames: ax.plot(ser.index,ser.values,"-o",color=c,lw=2.4,ms=6,label=lab)
    ax.axhline(1.0,color="#888",ls="--",lw=1); ax.text(ax.get_xlim()[1],1.02,"FP16 = FP32",ha="right",fontsize=9,color="#888")
    ax.set_xscale("log",base=2); ax.set_xticks(ser.index); ax.set_xticklabels(ser.index)
    ax.set_xlabel("Channels (Cin = Cout)"); ax.set_ylabel("FP32 / FP16  (>1 = FP16 faster)")
    title(ax,"When does FP16 win? Tensor-Core speedup vs channels","NVIDIA L4")
    ax.legend(frameon=False); save(fig,"04_fp16_speedup"); made+=1

if mc is not None:
    row=mc[mc["C"]==ROOF_C].set_index("method")["conv_ms"]
    order=[m for m in ["naive","tiled","coarse","im2col","cuDNN","cuDNN16"] if m in row.index]
    fig,ax=plt.subplots(figsize=(8.4,4.8)); despine(ax)
    y=np.arange(len(order)); vals=[row[m] for m in order]
    ax.barh(y,vals,color=[CLR[m] for m in order]); ax.set_yticks(y); ax.set_yticklabels([LBL[m] for m in order]); ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlabel("Conv time (ms, log) — lower better")
    for yi,v in zip(y,vals): ax.text(v,yi,f" {v:.3f}",va="center",fontsize=9)
    title(ax,f"The optimization ladder (C={ROOF_C})","naive → tiled → coarsened → im2col+GEMM → cuDNN")
    save(fig,"05_ladder_bar"); made+=1

if fusion is not None:
    r=fusion[fusion["C"]==ROOF_C].set_index("metric")["ms"]
    if all(k in r.index for k in ["epilogue_unfused","epilogue_fused","full_unfused","full_fused"]):
        fig,(a1,a2)=plt.subplots(1,2,figsize=(9.5,4.6))
        for ax,(u,f,ttl) in [(a1,("epilogue_unfused","epilogue_fused","BN + ReLU (epilogue)")),
                             (a2,("full_unfused","full_fused","Conv + BN + ReLU (full)"))]:
            despine(ax); vals=[r[u],r[f]]; sp=r[u]/r[f]
            ax.bar(["unfused","fused"],vals,color=["#E4572E","#3AA655"])
            for i,v in enumerate(vals): ax.text(i,v,f"{v:.3f}",ha="center",va="bottom",fontsize=10)
            ax.set_title(f"{ttl}\n{sp:.2f}× from fusion",fontsize=11); ax.set_ylabel("ms")
        fig.suptitle(f"Kernel fusion removes memory round-trips (C={ROOF_C})",fontsize=13,fontweight="bold",y=1.02)
        save(fig,"06_fusion"); made+=1

if coarsen is not None:
    fig,ax=plt.subplots(figsize=(8,5)); despine(ax)
    for C in sorted(coarsen["C"].unique()):
        d=coarsen[coarsen["C"]==C].sort_values("coarsen")
        base=d[d["coarsen"]==1]["conv_ms"].values
        speed=(base[0]/d["conv_ms"].values) if len(base) else d["conv_ms"].values
        ax.plot(d["coarsen"],speed,"-o",lw=2.4,ms=7,label=f"C={C}")
    ax.set_xscale("log",base=2); ax.set_xticks([1,2,4,8]); ax.set_xticklabels([1,2,4,8])
    ax.set_xlabel("Coarsening factor (output channels per thread)"); ax.set_ylabel("Speedup vs COARSEN=1")
    title(ax,"Bonus: output-channel coarsening vs speedup","256×256 · 3×3 · NVIDIA L4")
    ax.legend(frameon=False); save(fig,"07_coarsen_sweep"); made+=1

if trt is not None:
    fig,ax=plt.subplots(figsize=(8,5.2)); despine(ax)
    p=trt.pivot(index="C",columns="impl",values="ms")
    for imp in ["pytorch","trt_fp32","trt_fp16"]:
        if imp in p.columns: ax.plot(p.index,p[imp],"-o",color=CLR[imp],lw=2.4,ms=6,label=LBL[imp])
    ax.set_xscale("log",base=2); ax.set_yscale("log"); ax.set_xticks(p.index); ax.set_xticklabels(p.index)
    ax.set_xlabel("Channels (Cin = Cout)"); ax.set_ylabel("Full-pipeline latency (ms, log)")
    title(ax,"TensorRT end-to-end: PyTorch → ONNX → TRT","Conv+BN+ReLU+Pool · N=1 · NVIDIA L4")
    ax.legend(frameon=False); save(fig,"08_tensorrt"); made+=1

fig,(a1,a2)=plt.subplots(1,2,figsize=(10,4.8))
despine(a1); b=a1.bar(["Naive","Tiled"],[SC_L1_MB["naive"],SC_L1_MB["tiled"]],color=["#E4572E","#3AA655"])
a1.set_ylabel("L1 / TEX traffic (MB)"); a1.set_title("On-chip traffic")
for bi,v in zip(b,[SC_L1_MB["naive"],SC_L1_MB["tiled"]]): a1.text(bi.get_x()+bi.get_width()/2,v,f"{v:.1f}",ha="center",va="bottom",fontweight="bold")
r_=SC_L1_MB["naive"]/SC_L1_MB["tiled"]
a1.annotate(f"{r_:.1f}× less",xy=(1,SC_L1_MB['tiled']),xytext=(0.5,SC_L1_MB['naive']*0.5),fontweight="bold",arrowprops=dict(arrowstyle="->"))
despine(a2); b=a2.bar(["Naive","Tiled"],[SC_DRAM_MB["naive"],SC_DRAM_MB["tiled"]],color=["#E4572E","#3AA655"])
a2.set_ylabel("DRAM read (MB)"); a2.set_title("Off-chip (DRAM) traffic"); a2.set_ylim(0,max(SC_DRAM_MB.values())*1.5)
for bi,v in zip(b,[SC_DRAM_MB["naive"],SC_DRAM_MB["tiled"]]): a2.text(bi.get_x()+bi.get_width()/2,v,f"{v:.2f}",ha="center",va="bottom",fontweight="bold")
a2.text(0.5,0.9,"≈ equal → win is on-chip,\nnot DRAM bandwidth",transform=a2.transAxes,ha="center",fontsize=9,style="italic",color="#555")
fig.suptitle("Why tiling wins: 43× less on-chip traffic, same DRAM (11×11)",fontsize=13,fontweight="bold",y=1.02)
save(fig,"09_profiling"); made+=1

if mc is not None:
    row=mc[mc["C"]==ROOF_C].set_index("method")["conv_ms"]; flops=conv_flops(ROOF_C)
    fig,ax=plt.subplots(figsize=(8.4,5.6)); despine(ax)
    ai=np.logspace(-1,3,200); ax.plot(ai,np.minimum(L4_FP32_TFLOPS*1e3+0*ai, L4_BW_GBs*ai),color="#333",lw=2)
    ax.axhline(L4_FP32_TFLOPS*1e3,color="#333",lw=2,ls=":")
    ax.text(ai[-1],L4_FP32_TFLOPS*1e3*1.05,f"FP32 peak {L4_FP32_TFLOPS:.0f} TFLOP/s",ha="right",fontsize=8,color="#333")
    for m in ["naive","tiled","coarse","im2col"]:
        if m not in row.index: continue
        gf=flops/(row[m]/1e3)/1e9
        if m in MC_DRAM_MB: inten=flops/(MC_DRAM_MB[m]*1e6)
        else:
            H=256;Ho=H-2; minb=(ROOF_C*H*H + ROOF_C*ROOF_C*9 + ROOF_C*Ho*Ho)*4; inten=flops/minb
        ax.scatter(inten,gf,s=120,color=CLR[m],zorder=5); ax.annotate(LBL[m],(inten,gf),textcoords="offset points",xytext=(8,4),fontsize=9)
    if "cuDNN" in row.index:
        g=flops/(row["cuDNN"]/1e3)/1e9; ax.axhline(g,color="#4C78A8",ls="--",lw=1.5)
        ax.text(ai[0]*1.2,g*1.06,f"cuDNN achieves {g/1e3:.1f} TFLOP/s",fontsize=8,color="#4C78A8")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP / byte)"); ax.set_ylabel("Achieved GFLOP/s")
    title(ax,f"Roofline: the custom-kernel journey (C={ROOF_C})","points move up-right as reuse improves")
    save(fig,"10_roofline"); made+=1

fig,ax=plt.subplots(figsize=(6.5,5)); despine(ax)
pred, meas = 44.0, SC_L1_MB["naive"]/SC_L1_MB["tiled"]
b=ax.bar(["Predicted\n(analytical)","Measured\n(Nsight)"],[pred,meas],color=["#999","#3AA655"],width=0.6)
for bi,v in zip(b,[pred,meas]): ax.text(bi.get_x()+bi.get_width()/2,v,f"{v:.1f}×",ha="center",va="bottom",fontsize=13,fontweight="bold")
ax.set_ylabel("L1 traffic reduction (naive ÷ tiled)"); ax.set_ylim(0,max(pred,meas)*1.25)
title(ax,"Model vs measured: the tiling win","first-principles prediction confirmed by the profiler")
save(fig,"11_model_vs_measured"); made+=1

if mc is not None:
    row=mc[mc["C"]==ROOF_C].set_index("method")["conv_ms"]; flops=conv_flops(ROOF_C)
    order=[m for m in ["naive","tiled","coarse","im2col","cuDNN"] if m in row.index]
    g=[flops/(row[m]/1e3)/1e9/1e3 for m in order]
    fig,ax=plt.subplots(figsize=(8.4,4.8)); despine(ax)
    ax.bar([LBL[m] for m in order],g,color=[CLR[m] for m in order])
    ax.axhline(L4_FP32_TFLOPS,color="#333",ls=":",lw=1.5); ax.text(len(order)-0.5,L4_FP32_TFLOPS*1.02,f"FP32 peak {L4_FP32_TFLOPS:.0f}",ha="right",fontsize=8)
    for i,v in enumerate(g): ax.text(i,v,f"{v:.1f}",ha="center",va="bottom",fontsize=9)
    ax.set_ylabel("Achieved TFLOP/s")
    title(ax,f"Compute efficiency: how close to peak (C={ROOF_C})","fraction of the L4's FP32 roof each implementation reaches")
    save(fig,"12_gflops"); made+=1

print(f"\nDONE — {made}/12 charts in {OUT}/")
