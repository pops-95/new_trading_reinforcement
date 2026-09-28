#!/usr/bin/env python3
"""CUDA-assisted streaming BANKNIFTY candidate builder.

Input:
  data/banknifty_cleaned/banknifty_option_features_clean.parquet

Output:
  data/banknifty_candidates/candidate_dataset.parquet
  data/banknifty_candidates/candidate_quality_report.json
  data/banknifty_candidates/candidate_feature_manifest.json

Features are causal (known at completed minute t). Diagnostic labels are prefixed
`label_`, begin from t+1, and keep the selected option contract fixed.
"""
from __future__ import annotations

import argparse, gc, json, os, shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import cupy as cp
except Exception as exc:
    raise SystemExit(f"CuPy required (e.g. cupy-cuda12x): {exc}")
try:
    import duckdb
except Exception as exc:
    raise SystemExit(f"DuckDB required: pip install duckdb\n{exc}")

HORIZONS=(10,20,30)
TARGETS=(20.0,30.0,40.0)
STOP=20.0

UNDERLYING=[
    "timestamp","spot_close","return_1m","return_3m","return_5m",
    "velocity_3m","velocity_5m","acceleration_1m","atr_14","atr_ratio_5_20",
    "path_efficiency_5","extension_ema20_atr","same_direction_sign",
    "same_direction_run_length","htf_5m_direction","htf_15m_direction",
    "htf_5m_return_1","htf_15m_return_1","htf_5m_bars_since_direction_change",
    "htf_15m_bars_since_direction_change"
]
CHAIN=[
    "pcr_oi","pcr_volume","ce_oi_total","pe_oi_total","ce_oi_center","pe_oi_center",
    "call_oi_center_change","put_oi_center_change","oi_migration_score",
    "chain_price_oi_bias","chain_oi_available"
]
OPTION=[
    "groww_symbol","strike","expiry_date","option_type","close","volume","open_interest",
    "option_return_1m","option_return_3m","option_return_5m","premium_change_1m",
    "premium_acceleration","volume_change_1m","volume_accel_3_10","volume_accel_1_10",
    "oi_change_1m","oi_pct_change_1m","price_oi_product","price_oi_change",
    "price_return_x_oi_pct_change","implied_volatility","iv_change_1m","iv_change_3m",
    "delta","delta_change_1m","gamma","theta_per_day","vega_per_pct","theta_burden",
    "gamma_responsiveness","moneyness","abs_moneyness","dte","momentum_maturity_score",
    "exhaustion_score","greeks_available","individual_oi_available","option_market_available",
    "liquidity_available","true_1m_prev_available"
]

def atomic_json(obj:Any,p:Path):
    p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+".tmp")
    t.write_text(json.dumps(obj,indent=2,default=str)); t.replace(p)

def atomic_parquet(df:pd.DataFrame,p:Path):
    p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+".tmp")
    df.to_parquet(t,index=False,compression="zstd"); t.replace(p)

def release():
    gc.collect()
    try:
        cp.get_default_memory_pool().free_all_blocks(); cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception: pass

def cuda_info():
    d=cp.cuda.Device(); pr=cp.cuda.runtime.getDeviceProperties(d.id); f,t=cp.cuda.runtime.memGetInfo()
    n=pr.get("name",b"unknown"); n=n.decode() if isinstance(n,bytes) else str(n)
    return {"name":n,"cc":f"{pr.get('major',0)}.{pr.get('minor',0)}","free_gb":f/2**30,"total_gb":t/2**30}

def dbconn(threads:int,mem:float,tmp:Path):
    tmp.mkdir(parents=True,exist_ok=True); c=duckdb.connect(); c.execute(f"PRAGMA threads={threads}")
    c.execute("PRAGMA preserve_insertion_order=false"); c.execute(f"PRAGMA memory_limit='{mem:.1f}GB'")
    c.execute(f"PRAGMA temp_directory='{str(tmp).replace(chr(39),chr(39)*2)}'"); return c

def gpu_rank(df,delta_min,delta_max,strike_step):
    if df.empty:return np.array([],dtype=np.float32)
    side=df.option_type.to_numpy(); call=cp.asarray(side=="CE")
    spot=cp.asarray(df.spot_close.to_numpy(float)); strike=cp.asarray(df.strike.to_numpy(float))
    vol=cp.asarray(df.volume.to_numpy(float)); oi=cp.asarray(df.open_interest.to_numpy(float))
    delta=cp.asarray(df.delta.to_numpy(float)); dte=cp.asarray(df.dte.to_numpy(float))
    mom=cp.asarray(df.momentum_maturity_score.to_numpy(float)); exh=cp.asarray(df.exhaustion_score.to_numpy(float))
    ad=cp.abs(delta); mid=(delta_min+delta_max)/2; half=max((delta_max-delta_min)/2,1e-6)
    inband=cp.isfinite(ad)&(ad>=delta_min)&(ad<=delta_max)
    ds=cp.where(inband,1-cp.minimum(cp.abs(ad-mid)/half,1),cp.where(cp.isfinite(ad),cp.maximum(0,0.5-cp.abs(ad-mid)),0))
    atm=cp.rint(spot/strike_step)*strike_step
    itm=cp.where(call,(strike<=atm)&(strike>=atm-strike_step),(strike>=atm)&(strike<=atm+strike_step))
    ss=cp.where(itm,1,cp.maximum(0,1-0.35*cp.abs(strike-atm)/strike_step))
    vs=cp.tanh(cp.log1p(cp.maximum(cp.nan_to_num(vol,nan=0),0))/8)
    os_=cp.tanh(cp.log1p(cp.maximum(cp.nan_to_num(oi,nan=0),0))/10)
    dtes=cp.where(cp.isfinite(dte),cp.exp(-cp.maximum(dte,0)/5),0)
    ms=cp.where(cp.isfinite(mom),cp.clip(mom/100,0,1),0.5)
    es=cp.where(cp.isfinite(exh),1-cp.clip(exh/100,0,1),0.5)
    score=3*ds+2*ss+1.3*vs+0.5*os_+dtes+0.8*ms+0.5*es
    out=cp.asnumpy(score).astype(np.float32); release(); return out

def prefilter(src:Path,dst:Path,max_dte,threads,mem,tmp):
    c=dbconn(threads,mem,tmp); s=str(src).replace("'","''"); d=str(dst).replace("'","''")
    c.execute(f"""COPY (SELECT * FROM read_parquet('{s}') WHERE option_type IN ('CE','PE') AND close>0 AND spot_close>0 AND strike>0 AND (dte IS NULL OR (dte>=0 AND dte<={max_dte})) AND (delta IS NULL OR ABS(delta) BETWEEN 0.20 AND 0.85)) TO '{d}' (FORMAT PARQUET,COMPRESSION ZSTD)"""); c.close()

def select_contracts(filtered:Path,outdir:Path,batch,delta_min,delta_max,strike_step):
    outdir.mkdir(parents=True,exist_ok=True); pf=pq.ParquetFile(filtered); carry=pd.DataFrame(); part=0
    for rb in pf.iter_batches(batch_size=batch):
        df=rb.to_pandas(); df=pd.concat([carry,df],ignore_index=True) if not carry.empty else df
        df["timestamp"]=pd.to_datetime(df.timestamp,errors="coerce"); df=df.dropna(subset=["timestamp"])
        if df.empty: continue
        last=df.timestamp.max(); carry=df[df.timestamp==last].copy(); work=df[df.timestamp!=last].copy()
        if work.empty: continue
        work["candidate_selection_score"]=gpu_rank(work,delta_min,delta_max,strike_step)
        idx=work.groupby(["timestamp","option_type"],sort=False).candidate_selection_score.idxmax()
        sel=work.loc[idx].sort_values(["timestamp","option_type"]); atomic_parquet(sel,outdir/f"selected_{part:05d}.parquet")
        print(f"selection {part:05d}: {len(sel):,} rows",flush=True); part+=1; del df,work,sel,rb; release()
    if not carry.empty:
        carry["candidate_selection_score"]=gpu_rank(carry,delta_min,delta_max,strike_step)
        idx=carry.groupby(["timestamp","option_type"],sort=False).candidate_selection_score.idxmax()
        atomic_parquet(carry.loc[idx],outdir/f"selected_{part:05d}.parquet")

def build_wide(glob:str,dst:Path,threads,mem,tmp):
    c=dbconn(threads,mem,tmp); g=glob.replace("'","''"); d=str(dst).replace("'","''")
    cols=c.execute(f"DESCRIBE SELECT * FROM read_parquet('{g}')").df().column_name.tolist(); aset=set(cols)
    common=[x for x in UNDERLYING+CHAIN if x in aset]; opts=[x for x in OPTION+["candidate_selection_score"] if x in aset]
    expr=["timestamp"]+[f"ANY_VALUE({x}) AS {x}" for x in common if x!="timestamp"]
    for side,pfx in (("CE","ce"),("PE","pe")):
        expr += [f"MAX(CASE WHEN option_type='{side}' THEN {x} ELSE NULL END) AS {pfx}_{x}" for x in opts]
    c.execute(f"COPY (SELECT {','.join(expr)} FROM read_parquet('{g}') GROUP BY timestamp ORDER BY timestamp) TO '{d}' (FORMAT PARQUET,COMPRESSION ZSTD)"); c.close()

def score_row(r,pfx):
    sign=1 if pfx=="ce" else -1; vals=[]
    def add(cond,w):
        if cond is not None: vals.append((1 if cond else 0,w))
    def v(n): return r.get(n,np.nan)
    for n,w,fn in [
        ("htf_5m_direction",15,lambda z:z*sign>0),("htf_15m_direction",10,lambda z:z*sign>=0),
        ("velocity_3m",12,lambda z:z*sign>0),("acceleration_1m",10,lambda z:z*sign>0),
        ("path_efficiency_5",10,lambda z:z>0.55),("extension_ema20_atr",8,lambda z:z*sign<1.5)]:
        z=v(n); add(fn(z),w) if pd.notna(z) else None
    for n,w,fn in [
        (f"{pfx}_premium_acceleration",10,lambda z:z>0),(f"{pfx}_volume_accel_1_10",8,lambda z:z>1.2),
        (f"{pfx}_option_return_3m",7,lambda z:z>0),(f"{pfx}_iv_change_3m",5,lambda z:z>=-0.05)]:
        z=v(n); add(fn(z),w) if pd.notna(z) else None
    return np.nan if not vals else 100*sum(a*w for a,w in vals)/sum(w for _,w in vals)

def exhaustion_row(r,pfx):
    sign=1 if pfx=="ce" else -1; vals=[]
    def add(cond,w): vals.append((1 if cond else 0,w))
    checks=[
        ("extension_ema20_atr",25,lambda z:z*sign>1.5),
        ("acceleration_1m",20,lambda z:z*sign<0),
        ("atr_ratio_5_20",15,lambda z:z>1.5),
        (f"{pfx}_option_return_3m",15,lambda z:z<=0),
        (f"{pfx}_iv_change_3m",10,lambda z:z>0.10)]
    for n,w,fn in checks:
        z=r.get(n,np.nan); add(fn(z),w) if pd.notna(z) else None
    run=r.get("same_direction_run_length",np.nan); s=r.get("same_direction_sign",np.nan)
    if pd.notna(run) and pd.notna(s): add(run>=5 and s==sign,15)
    return np.nan if not vals else 100*sum(a*w for a,w in vals)/sum(w for _,w in vals)

def score_states(src:Path,parts:Path,batch):
    if parts.exists(): shutil.rmtree(parts)
    parts.mkdir(parents=True); pf=pq.ParquetFile(src)
    for i,rb in enumerate(pf.iter_batches(batch_size=batch)):
        df=rb.to_pandas();
        df["ce_momentum_score"]=[score_row(r,"ce") for _,r in df.iterrows()]; df["pe_momentum_score"]=[score_row(r,"pe") for _,r in df.iterrows()]
        df["ce_exhaustion_score_final"]=[exhaustion_row(r,"ce") for _,r in df.iterrows()]; df["pe_exhaustion_score_final"]=[exhaustion_row(r,"pe") for _,r in df.iterrows()]
        for pfx in ("ce","pe"):
            sym=df.get(f"{pfx}_groww_symbol",pd.Series(index=df.index,dtype=object)); close=df.get(f"{pfx}_close",pd.Series(index=df.index,dtype=float))
            df[f"{pfx}_candidate_available"]=(sym.notna()&close.gt(0)).astype("int8")
            df[f"{pfx}_momentum_eligible"]=(df[f"{pfx}_candidate_available"].eq(1)&df[f"{pfx}_momentum_score"].ge(55)&(df[f"{pfx}_exhaustion_score_final"].isna()|df[f"{pfx}_exhaustion_score_final"].le(70))).astype("int8")
        atomic_parquet(df,parts/f"state_{i:05d}.parquet"); print(f"state {i:05d}: {len(df):,}",flush=True); del df,rb; release()

def combine(glob,dst,threads,mem,tmp):
    c=dbconn(threads,mem,tmp); g=str(glob).replace("'","''"); d=str(dst).replace("'","''")
    c.execute(f"COPY (SELECT * FROM read_parquet('{g}') ORDER BY timestamp) TO '{d}' (FORMAT PARQUET,COMPRESSION ZSTD)"); c.close()

def gpu_labels(ts_ns,prem,horizons=HORIZONS,targets=TARGETS,stop=STOP):
    n=len(prem); out={}
    for h in horizons: out[f"mfe_{h}m"]=np.full(n,np.nan,np.float32); out[f"mae_{h}m"]=np.full(n,np.nan,np.float32)
    for t in targets: out[f"hit_{int(t)}_before_stop"]=np.full(n,np.nan,np.float32); out[f"time_to_{int(t)}"]=np.full(n,np.nan,np.float32)
    out["time_to_stop"]=np.full(n,np.nan,np.float32)
    if n<2:return out
    p=cp.asarray(prem,dtype=cp.float64); ts=cp.asarray(ts_ns,dtype=cp.int64); one=int(pd.Timedelta(minutes=1).value); maxh=max(horizons)
    entry=cp.full(n,cp.nan); entry[:-1]=p[1:]; ve=cp.zeros(n,dtype=cp.bool_); ve[:-1]=cp.isfinite(p[:-1])&cp.isfinite(p[1:])&((ts[1:]-ts[:-1])==one)
    for h in horizons:
        mx=cp.full(n,cp.nan); mn=cp.full(n,cp.nan)
        for k in range(1,h+1):
            m=n-k
            if m<=0: break
            fut=p[k:]; base=entry[:m]; valid=ve[:m]&((ts[k:]-ts[:m])==k*one)&cp.isfinite(fut)&cp.isfinite(base); mv=fut-base
            ex=mx[:m]; mx[:m]=cp.where(valid,cp.where(cp.isnan(ex),mv,cp.maximum(ex,mv)),ex); ex=mn[:m]; mn[:m]=cp.where(valid,cp.where(cp.isnan(ex),mv,cp.minimum(ex,mv)),ex)
        out[f"mfe_{h}m"]=cp.asnumpy(mx).astype(np.float32); out[f"mae_{h}m"]=cp.asnumpy(mn).astype(np.float32)
    fs=cp.full(n,-1,dtype=cp.int32); ft={t:cp.full(n,-1,dtype=cp.int32) for t in targets}
    for k in range(1,maxh+1):
        m=n-k
        if m<=0: break
        fut=p[k:]; base=entry[:m]; valid=ve[:m]&((ts[k:]-ts[:m])==k*one)&cp.isfinite(fut)&cp.isfinite(base); mv=fut-base
        cur=fs[:m]; fs[:m]=cp.where((cur<0)&valid&(mv<=-stop),k,cur)
        for t in targets:
            cur=ft[t][:m]; ft[t][:m]=cp.where((cur<0)&valid&(mv>=t),k,cur)
    fsc=cp.asnumpy(fs); out["time_to_stop"]=np.where(fsc>=0,fsc,np.nan).astype(np.float32)
    for t in targets:
        f=cp.asnumpy(ft[t]); out[f"time_to_{int(t)}"]=np.where(f>=0,f,np.nan).astype(np.float32); out[f"hit_{int(t)}_before_stop"]=np.where(f>=0,np.where((fsc<0)|(f<fsc),1.0,0.0),np.where(fsc>=0,0.0,np.nan)).astype(np.float32)
    release(); return out

def label_side(clean:Path,candidate:Path,pfx:str,outdir:Path):
    if outdir.exists(): shutil.rmtree(outdir)
    outdir.mkdir(parents=True); cand=pd.read_parquet(candidate,columns=["timestamp",f"{pfx}_groww_symbol"]); cand=cand.dropna(subset=[f"{pfx}_groww_symbol"])
    by={s:g[["timestamp"]].sort_values("timestamp") for s,g in cand.groupby(f"{pfx}_groww_symbol",sort=False)}; c=duckdb.connect(); src=str(clean).replace("'","''"); part=0
    for sym,wanted in by.items():
        ss=str(sym).replace("'","''"); px=c.execute(f"SELECT timestamp,close FROM read_parquet('{src}') WHERE groww_symbol='{ss}' ORDER BY timestamp").df()
        if px.empty: continue
        px["timestamp"]=pd.to_datetime(px.timestamp,errors="coerce"); px=px.dropna().drop_duplicates("timestamp",keep="last").sort_values("timestamp")
        lab=gpu_labels(px.timestamp.to_numpy(dtype="datetime64[ns]").astype("int64"),px.close.to_numpy(float)); ldf=pd.DataFrame({"timestamp":px.timestamp,**{f"label_{pfx}_{k}":v for k,v in lab.items()}})
        m=wanted.merge(ldf,on="timestamp",how="left"); m[f"{pfx}_groww_symbol"]=sym; atomic_parquet(m,outdir/f"{pfx}_{part:05d}.parquet"); part+=1
        if part%100==0: print(f"{pfx.upper()} labels: {part}",flush=True)
        del px,ldf,m; release()
    c.close(); del cand,by; release()

def merge_labels(cand,ceglob,peglob,dst,threads,mem,tmp):
    c=dbconn(threads,mem,tmp); a=str(cand).replace("'","''"); ce=str(ceglob).replace("'","''"); pe=str(peglob).replace("'","''"); d=str(dst).replace("'","''")
    c.execute(f"""COPY (SELECT c.*, ce.* EXCLUDE(timestamp,ce_groww_symbol), pe.* EXCLUDE(timestamp,pe_groww_symbol) FROM read_parquet('{a}') c LEFT JOIN read_parquet('{ce}') ce ON c.timestamp=ce.timestamp AND c.ce_groww_symbol=ce.ce_groww_symbol LEFT JOIN read_parquet('{pe}') pe ON c.timestamp=pe.timestamp AND c.pe_groww_symbol=pe.pe_groww_symbol ORDER BY c.timestamp) TO '{d}' (FORMAT PARQUET,COMPRESSION ZSTD)"""); c.close()

def report(final, outdir):
    c = duckdb.connect()

    s = str(final).replace("'", "''")

    r = c.execute(f"""
        SELECT
            COUNT(*) AS row_count,

            AVG(
                CAST(ce_candidate_available AS DOUBLE)
            ) AS ce_available_fraction,

            AVG(
                CAST(pe_candidate_available AS DOUBLE)
            ) AS pe_available_fraction,

            AVG(
                CAST(ce_momentum_eligible AS DOUBLE)
            ) AS ce_eligible_fraction,

            AVG(
                CAST(pe_momentum_eligible AS DOUBLE)
            ) AS pe_eligible_fraction,

            AVG(
                CAST(chain_oi_available AS DOUBLE)
            ) AS chain_oi_available_fraction

        FROM read_parquet('{s}')
    """).df().iloc[0].to_dict()

    c.close()

    r = {
        k: (
            int(v)
            if k == "row_count"
            else float(v)
            if pd.notna(v)
            else None
        )
        for k, v in r.items()
    }

    atomic_json(
        r,
        outdir / "candidate_quality_report.json"
    )

    return r

def manifest(final:Path,outdir:Path):
    sch=pq.ParquetFile(final).schema_arrow; m={}
    for f in sch:
        role="diagnostic_label" if f.name.startswith("label_") else "identifier" if f.name in {"timestamp","ce_groww_symbol","pe_groww_symbol","ce_expiry_date","pe_expiry_date"} else "feature"
        m[f.name]={"dtype":str(f.type),"role":role,"allowed_in_rl_observation":role=="feature"}
    atomic_json(m,outdir/"candidate_feature_manifest.json")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input-file",default="data/banknifty_cleaned/banknifty_option_features_clean.parquet"); ap.add_argument("--output-dir",default="data/banknifty_candidates")
    ap.add_argument("--batch-rows",type=int,default=150000); ap.add_argument("--delta-min",type=float,default=0.45); ap.add_argument("--delta-max",type=float,default=0.60); ap.add_argument("--strike-step",type=float,default=100.0); ap.add_argument("--max-dte",type=float,default=10.0)
    ap.add_argument("--duckdb-threads",type=int,default=4); ap.add_argument("--duckdb-memory-gb",type=float,default=4.0); ap.add_argument("--skip-labels",action="store_true"); ap.add_argument("--fresh",action="store_true"); a=ap.parse_args()
    src=Path(a.input_file); out=Path(a.output_dir)
    if not src.exists(): raise SystemExit(f"Missing input: {src}")
    if a.fresh and out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True); tmp=out/"duckdb_temp"; dev=cuda_info(); print(f"CUDA: {dev['name']} CC={dev['cc']} free={dev['free_gb']:.2f}GB",flush=True); atomic_json(dev,out/"cuda_info.json")
    filtered=out/"candidate_source_filtered.parquet"; selected=out/"selected_parts"; wide=out/"candidate_state_raw.parquet"; scoredparts=out/"scored_parts"; scored=out/"candidate_state_scored.parquet"; final=out/"candidate_dataset.parquet"
    print("1/7 prefilter",flush=True); prefilter(src,filtered,a.max_dte,a.duckdb_threads,a.duckdb_memory_gb,tmp) if not filtered.exists() else print(" cached")
    print("2/7 CUDA contract ranking",flush=True); select_contracts(filtered,selected,a.batch_rows,a.delta_min,a.delta_max,a.strike_step) if not any(selected.glob("selected_*.parquet")) else print(" cached")
    print("3/7 wide CE/PE state",flush=True); build_wide(str(selected/"selected_*.parquet"),wide,a.duckdb_threads,a.duckdb_memory_gb,tmp) if not wide.exists() else print(" cached")
    print("4/7 state scores",flush=True)
    if not scored.exists(): score_states(wide,scoredparts,a.batch_rows); combine(scoredparts/"state_*.parquet",scored,a.duckdb_threads,a.duckdb_memory_gb,tmp)
    else: print(" cached")
    if a.skip_labels:
        shutil.copy2(scored,final); print("5/7 labels skipped"); r=report(final,out); manifest(final,out); print("DONE",r); return
    ce=out/"ce_label_parts"; pe=out/"pe_label_parts"; print("5/7 CUDA future labels",flush=True)
    if not any(ce.glob("*.parquet")): label_side(src,scored,"ce",ce)
    else: print(" CE cached")
    if not any(pe.glob("*.parquet")): label_side(src,scored,"pe",pe)
    else: print(" PE cached")
    print("6/7 merge labels",flush=True); merge_labels(scored,str(ce/"*.parquet"),str(pe/"*.parquet"),final,a.duckdb_threads,a.duckdb_memory_gb,tmp)
    print("7/7 report/manifest",flush=True); r=report(final,out); manifest(final,out); print("\nCANDIDATE DATASET COMPLETE"); print(json.dumps(r,indent=2)); print("Dataset:",final)

if __name__=="__main__": main()
