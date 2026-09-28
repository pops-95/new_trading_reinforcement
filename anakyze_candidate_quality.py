#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import cupy as cp
except Exception as e:
    raise SystemExit(f"CuPy required: {e}")

MOM_BINS=[0,50,60,70,80,90,100.000001]
MOM_LABELS=["0-50","50-60","60-70","70-80","80-90","90-100"]
EXH_BINS=[0,25,50,75,100.000001]
EXH_LABELS=["0-25","25-50","50-75","75-100"]
HORIZONS=(10,20,30)
TARGETS=(20,30,40)

def cuda_info():
    d=cp.cuda.Device(); p=cp.cuda.runtime.getDeviceProperties(d.id)
    n=p.get("name",b"unknown")
    if isinstance(n,bytes): n=n.decode(errors="ignore")
    free_b,total_b=cp.cuda.runtime.memGetInfo()
    return {"name":str(n),"cc":f"{int(p.get('major',0))}.{int(p.get('minor',0))}","free_gb":round(free_b/1024**3,3),"total_gb":round(total_b/1024**3,3)}

def release():
    gc.collect()
    try:
        cp.get_default_memory_pool().free_all_blocks(); cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception: pass

def gmean(a):
    if len(a)==0:return np.nan
    x=cp.asarray(a,dtype=cp.float64); v=cp.nanmean(x)
    out=float(cp.asnumpy(v)) if bool(cp.isfinite(v)) else np.nan
    del x,v; return out

def gmedian(a):
    if len(a)==0:return np.nan
    x=cp.asarray(a,dtype=cp.float64); x=x[cp.isfinite(x)]
    if x.size==0:return np.nan
    out=float(cp.asnumpy(cp.median(x))); del x; return out

def gcorr(a,b):
    x=cp.asarray(a,dtype=cp.float64); y=cp.asarray(b,dtype=cp.float64)
    m=cp.isfinite(x)&cp.isfinite(y)
    if int(m.sum())<3:return np.nan
    xx=x[m]; yy=y[m]
    if float(cp.std(xx))==0 or float(cp.std(yy))==0:return np.nan
    r=float(cp.asnumpy(cp.corrcoef(xx,yy)[0,1])); del x,y,m,xx,yy; return r

def bucket(a,bins):
    x=cp.asarray(a,dtype=cp.float64); b=cp.asarray(bins,dtype=cp.float64)
    idx=cp.digitize(x,b[1:-1],right=False); idx=cp.where(cp.isfinite(x),idx,-1)
    out=cp.asnumpy(idx).astype(np.int16); del x,b,idx; return out

def cols(side):
    p=side.lower()
    return {
        "momentum":f"{p}_momentum_score","exhaustion":f"{p}_exhaustion_score_final","eligible":f"{p}_momentum_eligible",
        "indoi":f"{p}_individual_oi_available","greeks":f"{p}_greeks_available",
        "hit20":f"label_{p}_hit_20_before_stop","hit30":f"label_{p}_hit_30_before_stop","hit40":f"label_{p}_hit_40_before_stop",
        "mfe10":f"label_{p}_mfe_10m","mfe20":f"label_{p}_mfe_20m","mfe30":f"label_{p}_mfe_30m",
        "mae10":f"label_{p}_mae_10m","mae20":f"label_{p}_mae_20m","mae30":f"label_{p}_mae_30m",
        "t20":f"label_{p}_time_to_20","t40":f"label_{p}_time_to_40","tstop":f"label_{p}_time_to_stop"
    }

def load_side(path,side,batch_rows):
    c=cols(side); pf=pq.ParquetFile(path); avail=set(pf.schema.names)
    wanted=[x for x in list(c.values())+["chain_oi_available"] if x in avail]
    critical=[c["momentum"],c["hit20"],c["hit30"],c["hit40"]]
    missing=[x for x in critical if x not in avail]
    if missing: raise ValueError(f"{side} missing critical columns: {missing}")
    chunks=[]
    for rb in pf.iter_batches(batch_size=batch_rows,columns=wanted): chunks.append(rb.to_pandas())
    return pd.concat(chunks,ignore_index=True) if chunks else pd.DataFrame()

def summarize(df,side):
    c=cols(side); r={"count":int(len(df))}
    for t in TARGETS:
        col=c[f"hit{t}"]; r[f"hit_{t}_before_stop_rate"]=gmean(df[col].to_numpy(float)) if col in df else np.nan
    for h in HORIZONS:
        for m in ("mfe","mae"):
            col=c[f"{m}{h}"]; a=df[col].to_numpy(float) if col in df else np.array([])
            r[f"{m}_{h}m_mean"]=gmean(a); r[f"{m}_{h}m_median"]=gmedian(a)
    for name,key in (("time_to_20","t20"),("time_to_40","t40"),("time_to_stop","tstop")):
        col=c[key]; a=df[col].to_numpy(float) if col in df else np.array([])
        r[f"{name}_mean"]=gmean(a); r[f"{name}_median"]=gmedian(a)
    r["eligibility_rate"]=gmean(df[c["eligible"]].to_numpy(float)) if c["eligible"] in df else np.nan
    return r

def momentum_table(df,side):
    c=cols(side); idx=bucket(df[c["momentum"]].to_numpy(float),MOM_BINS); base=summarize(df,side); rows=[]
    for i,lbl in enumerate(MOM_LABELS):
        s=summarize(df[idx==i],side); s["side"]=side.upper(); s["momentum_bucket"]=lbl
        for t in TARGETS:
            b=base[f"hit_{t}_before_stop_rate"]; v=s[f"hit_{t}_before_stop_rate"]
            s[f"hit_{t}_lift_vs_baseline"]=v/b if pd.notna(v) and pd.notna(b) and b!=0 else np.nan
        rows.append(s)
    return pd.DataFrame(rows)

def mex_table(df,side):
    c=cols(side)
    if c["exhaustion"] not in df:return pd.DataFrame()
    mi=bucket(df[c["momentum"]].to_numpy(float),MOM_BINS); ei=bucket(df[c["exhaustion"]].to_numpy(float),EXH_BINS); rows=[]
    for i,ml in enumerate(MOM_LABELS):
        for j,el in enumerate(EXH_LABELS):
            s=summarize(df[(mi==i)&(ei==j)],side); s["side"]=side.upper(); s["momentum_bucket"]=ml; s["exhaustion_bucket"]=el; rows.append(s)
    return pd.DataFrame(rows)

def availability_table(df,side):
    c=cols(side); groups=[]
    if "chain_oi_available" in df: groups.append(("chain_oi","chain_oi_available"))
    if c["indoi"] in df: groups.append(("individual_oi",c["indoi"]))
    if c["greeks"] in df: groups.append(("greeks",c["greeks"]))
    rows=[]
    for name,col in groups:
        for flag in (0,1):
            s=summarize(df[df[col].fillna(0).astype(int)==flag],side); s["side"]=side.upper(); s["availability_group"]=name; s["available"]=flag; rows.append(s)
    return pd.DataFrame(rows)

def separability(df,side):
    c=cols(side); mom=df[c["momentum"]].to_numpy(float); out={"rows":int(len(df))}
    for t in TARGETS: out[f"corr_momentum_vs_hit{t}"]=gcorr(mom,df[c[f"hit{t}"]].to_numpy(float))
    if c["exhaustion"] in df:
        exh=df[c["exhaustion"]].to_numpy(float)
        for t in TARGETS: out[f"corr_exhaustion_vs_hit{t}"]=gcorr(exh,df[c[f"hit{t}"]].to_numpy(float))
    valid=df[c["momentum"]].dropna()
    if len(valid):
        q80=float(valid.quantile(.8)); q90=float(valid.quantile(.9)); out["momentum_q80"]=q80; out["momentum_q90"]=q90; base=summarize(df,side)
        for name,q in (("top20",q80),("top10",q90)):
            ss=summarize(df[df[c["momentum"]]>=q],side)
            for t in TARGETS:
                b=base[f"hit_{t}_before_stop_rate"]; v=ss[f"hit_{t}_before_stop_rate"]; out[f"{name}_hit{t}_rate"]=v; out[f"{name}_hit{t}_lift"]=v/b if pd.notna(v) and pd.notna(b) and b!=0 else np.nan
    lifts=[out.get("top10_hit20_lift"),out.get("top10_hit30_lift"),out.get("top10_hit40_lift")]; lifts=[x for x in lifts if x is not None and np.isfinite(x)]
    out["separability_flag"]="INSUFFICIENT_LABEL_DATA" if not lifts else "MATERIAL_LIFT_PRESENT" if max(lifts)>=1.15 else "WEAK_LIFT_PRESENT" if max(lifts)>=1.05 else "LITTLE_OR_NO_LIFT"
    return out

def combined_table(ce,pe):
    rows=[]
    for side,df in (("ce",ce),("pe",pe)):
        c=cols(side); tmp=pd.DataFrame({"momentum":df[c["momentum"]]})
        for t in TARGETS: tmp[f"hit{t}"]=df[c[f"hit{t}"]]
        tmp["eligible"]=df[c["eligible"]] if c["eligible"] in df else np.nan
        for h in HORIZONS:
            tmp[f"mfe{h}"]=df[c[f"mfe{h}"]] if c[f"mfe{h}"] in df else np.nan; tmp[f"mae{h}"]=df[c[f"mae{h}"]] if c[f"mae{h}"] in df else np.nan
        tmp["t20"]=df[c["t20"]] if c["t20"] in df else np.nan; tmp["t40"]=df[c["t40"]] if c["t40"] in df else np.nan; tmp["tstop"]=df[c["tstop"]] if c["tstop"] in df else np.nan; rows.append(tmp)
    d=pd.concat(rows,ignore_index=True); idx=bucket(d["momentum"].to_numpy(float),MOM_BINS); out=[]
    for i,lbl in enumerate(MOM_LABELS):
        sub=d[idx==i]; r={"side":"COMBINED","momentum_bucket":lbl,"count":int(len(sub)),"eligibility_rate":gmean(sub["eligible"].to_numpy(float))}
        for t in TARGETS:r[f"hit_{t}_before_stop_rate"]=gmean(sub[f"hit{t}"].to_numpy(float))
        for h in HORIZONS:
            for m in ("mfe","mae"):
                a=sub[f"{m}{h}"].to_numpy(float); r[f"{m}_{h}m_mean"]=gmean(a); r[f"{m}_{h}m_median"]=gmedian(a)
        for name,col in (("time_to_20","t20"),("time_to_40","t40"),("time_to_stop","tstop")):
            a=sub[col].to_numpy(float); r[f"{name}_mean"]=gmean(a); r[f"{name}_median"]=gmedian(a)
        out.append(r)
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input-file",default="data/banknifty_candidates/candidate_dataset.parquet"); ap.add_argument("--output-dir",default="data/banknifty_candidate_analysis"); ap.add_argument("--batch-rows",type=int,default=150000); ap.add_argument("--side",choices=["both","ce","pe"],default="both"); args=ap.parse_args()
    inp=Path(args.input_file); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    if not inp.exists():raise SystemExit(f"Missing input: {inp}")
    dev=cuda_info(); print(f"CUDA device: {dev['name']} | CC={dev['cc']} | free={dev['free_gb']:.2f} GB")
    sides=["ce","pe"] if args.side=="both" else [args.side]; data={}; bfs=[]; mefs=[]; avfs=[]; summary={"cuda":dev,"input_file":str(inp),"sides":{}}
    for side in sides:
        print(f"Analyzing {side.upper()}..."); df=load_side(inp,side,args.batch_rows); data[side]=df; bfs.append(momentum_table(df,side)); me=mex_table(df,side); av=availability_table(df,side)
        if not me.empty:mefs.append(me)
        if not av.empty:avfs.append(av)
        summary["sides"][side.upper()]=separability(df,side); release()
    if "ce" in data and "pe" in data:bfs.append(combined_table(data["ce"],data["pe"]))
    pd.concat(bfs,ignore_index=True).to_csv(out/"momentum_bucket_analysis.csv",index=False)
    (pd.concat(mefs,ignore_index=True) if mefs else pd.DataFrame()).to_csv(out/"momentum_exhaustion_analysis.csv",index=False)
    (pd.concat(avfs,ignore_index=True) if avfs else pd.DataFrame()).to_csv(out/"availability_subgroup_analysis.csv",index=False)
    with open(out/"candidate_quality_summary.json","w",encoding="utf-8") as f:json.dump(summary,f,indent=2,default=str)
    print("\nCANDIDATE QUALITY ANALYSIS COMPLETE"); print("Summary:",out/"candidate_quality_summary.json"); print("Momentum buckets:",out/"momentum_bucket_analysis.csv"); print("Momentum x exhaustion:",out/"momentum_exhaustion_analysis.csv"); print("Availability subgroups:",out/"availability_subgroup_analysis.csv")

if __name__=="__main__":main()
