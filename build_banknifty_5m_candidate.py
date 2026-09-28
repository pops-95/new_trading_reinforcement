#!/usr/bin/env python3
# v2: safe CUDA future-horizon reductions; no All-NaN slice warnings
from __future__ import annotations
import argparse, gc, json, os, shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
os.environ.setdefault('OMP_NUM_THREADS','2'); os.environ.setdefault('OPENBLAS_NUM_THREADS','2'); os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import duckdb
import cupy as cp

TARGETS=(20.0,30.0,40.0); STOP=20.0; HORIZONS={15:3,30:6,45:9,60:12}
COMMON=['timestamp','spot_close','return_5m','return_10m','return_15m','return_30m','true_range_5m','atr14_5m','atr5_5m','atr20_5m','atr_expansion_5m','body_to_range_5m','close_location_5m','ema9_5m','ema20_5m','ema50_5m','ema9_distance_atr','ema20_distance_atr','ema50_distance_atr','ema9_slope_3bars','ema20_slope_3bars','ema50_slope_3bars','velocity_5m','velocity_10m','acceleration_5m','jerk_5m','path_efficiency_3bars','path_efficiency_6bars','day_open_5m','day_high_5m','day_low_5m','return_from_day_open','direction_sign_5m','direction_run_5m','return_15m_regime','ema_spread_15m','direction_15m','regime_15m_available','pcr_oi','pcr_volume','ce_oi_total','pe_oi_total','ce_volume_total','pe_volume_total','ce_oi_center','pe_oi_center','call_oi_center_change','put_oi_center_change','oi_migration_score','chain_price_oi_bias','chain_oi_available']
OPTION=['groww_symbol','strike','expiry_date','option_type','open','high','low','close','volume','open_interest','source_1m_rows','implied_volatility','delta','gamma','theta_per_day','vega_per_pct','theta_burden','gamma_responsiveness','dte','moneyness','abs_moneyness','option_return_5m','option_return_10m','option_return_15m','premium_change_5m','premium_acceleration_5m','volume_change_5m','volume_acceleration_5m','oi_change_5m','oi_pct_change_5m','oi_acceleration_5m','iv_change_5m','iv_acceleration_5m','delta_change_5m','option_body_to_range_5m','option_close_location_5m','price_oi_change_5m','price_return_x_oi_pct_5m','true_5m_prev_available','individual_oi_available_5m','greeks_available_5m']

def atomic_json(obj,p):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(obj,indent=2,default=str)); t.replace(p)
def atomic_parquet(df,p):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp'); df.to_parquet(t,index=False,compression='zstd'); t.replace(p)
def release():
 gc.collect(); cp.get_default_memory_pool().free_all_blocks(); cp.get_default_pinned_memory_pool().free_all_blocks()
def cuda_info():
 d=cp.cuda.Device(); pr=cp.cuda.runtime.getDeviceProperties(d.id); f,t=cp.cuda.runtime.memGetInfo(); n=pr.get('name',b'unknown'); n=n.decode(errors='ignore') if isinstance(n,bytes) else str(n)
 return {'name':n,'cc':f"{int(pr.get('major',0))}.{int(pr.get('minor',0))}",'free_gb':round(f/1024**3,3),'total_gb':round(t/1024**3,3)}
def db(threads,mem,temp):
 temp.mkdir(parents=True,exist_ok=True); c=duckdb.connect(); c.execute(f'PRAGMA threads={max(1,int(threads))}'); c.execute('PRAGMA preserve_insertion_order=false'); c.execute(f"PRAGMA memory_limit='{float(mem):.2f}GB'"); c.execute(f"PRAGMA temp_directory='{str(temp).replace(chr(39),chr(39)*2)}'"); return c
def q(p): return str(p).replace("'","''")
def dt(df):
 if 'timestamp' in df: df['timestamp']=pd.to_datetime(df['timestamp'],errors='coerce')
 if 'expiry_date' in df: df['expiry_date']=pd.to_datetime(df['expiry_date'],errors='coerce')
def clean(df):
 for c in df.columns:
  if pd.api.types.is_numeric_dtype(df[c]): df[c]=pd.to_numeric(df[c],errors='coerce').replace([np.inf,-np.inf],np.nan)

def source_mode(input_dir,optfile):
 if optfile:
  p=Path(optfile)
  if not p.exists(): raise SystemExit(f'Missing: {p}')
  return 'single',p
 ps=sorted((input_dir/'option_parts').glob('day_*.parquet'))
 if ps: return 'parts',ps
 p=input_dir/'banknifty_options_5m_features.parquet'
 if p.exists(): return 'single',p
 raise SystemExit('No 5m option source found')

def days_single(path,threads,mem,temp):
 c=db(threads,mem,temp); s=q(path); d=c.execute(f"SELECT DISTINCT CAST(timestamp AS DATE) d FROM read_parquet('{s}') ORDER BY d").df(); c.close(); return [pd.Timestamp(x) for x in d.d]
def extract_day(src,day,dst,threads,mem,temp):
 c=db(threads,mem,temp); c.execute(f"COPY (SELECT * FROM read_parquet('{q(src)}') WHERE CAST(timestamp AS DATE)=DATE '{day:%Y-%m-%d}') TO '{q(dst)}' (FORMAT PARQUET, COMPRESSION ZSTD)"); c.close()

def carr(df,col,default=np.nan):
 if col not in df: return cp.full(len(df),default,dtype=cp.float32)
 return cp.asarray(pd.to_numeric(df[col],errors='coerce').to_numpy(np.float32))
def rank_cuda(df,dmin,dmax,step):
 n=len(df); spot=carr(df,'spot_close'); strike=carr(df,'strike'); delta=cp.abs(carr(df,'delta')); vol=carr(df,'volume'); oi=carr(df,'open_interest'); dte=carr(df,'dte'); ret=carr(df,'option_return_5m'); acc=carr(df,'premium_acceleration_5m'); vacc=carr(df,'volume_acceleration_5m'); rows=carr(df,'source_1m_rows'); ce=cp.asarray(df.option_type.eq('CE').to_numpy())
 mid=(dmin+dmax)/2; half=max((dmax-dmin)/2,1e-5); band=cp.isfinite(delta)&(delta>=dmin)&(delta<=dmax); ds=cp.where(band,1-cp.minimum(cp.abs(delta-mid)/half,1),cp.where(cp.isfinite(delta),cp.maximum(0,0.5-cp.abs(delta-mid)),0))
 atm=cp.rint(spot/step)*step; dist=cp.abs(strike-atm)/step; itm=cp.where(ce,(strike<=atm)&(strike>=atm-step),(strike>=atm)&(strike<=atm+step)); ss=cp.where(itm,1,cp.maximum(0,1-0.35*dist)); vs=cp.tanh(cp.log1p(cp.maximum(cp.nan_to_num(vol,nan=0),0))/8); os=cp.tanh(cp.log1p(cp.maximum(cp.nan_to_num(oi,nan=0),0))/10); des=cp.where(cp.isfinite(dte),cp.exp(-cp.maximum(dte,0)/5),0); comp=cp.where(cp.isfinite(rows),cp.clip(rows/5,0,1),0); mb=cp.where(cp.isfinite(ret),cp.tanh(cp.maximum(ret,0)*4),0); ab=cp.where(cp.isfinite(acc),cp.tanh(cp.maximum(acc,0)/20),0); vb=cp.where(cp.isfinite(vacc),cp.tanh(cp.maximum(vacc-1,0)),0)
 out=cp.asnumpy(3*ds+2*ss+1.2*vs+0.35*os+des+0.75*comp+0.35*mb+0.25*ab+0.2*vb).astype(np.float32); release(); return out

def side_scores(w,side):
 p=side; n=len(w); sign=1.0 if p=='ce' else -1.0
 def a(c): return carr(w,c)
 total=cp.zeros(n,dtype=cp.float32); weight=cp.zeros(n,dtype=cp.float32)
 def add(cond,av,wt):
  nonlocal total,weight; total+=cp.where(av,cond.astype(cp.float32)*wt,0); weight+=cp.where(av,wt,0)
 vel=a('velocity_5m'); vel10=a('velocity_10m'); acc=a('acceleration_5m'); path=a('path_efficiency_3bars'); atr=a('atr_expansion_5m'); htf=a('direction_15m'); o5=a(f'{p}_option_return_5m'); o10=a(f'{p}_option_return_10m'); pa=a(f'{p}_premium_acceleration_5m'); va=a(f'{p}_volume_acceleration_5m'); iv=a(f'{p}_iv_change_5m'); dc=a(f'{p}_delta_change_5m')
 for cond,av,wt in [(vel*sign>0,cp.isfinite(vel),12),(vel10*sign>0,cp.isfinite(vel10),10),(acc*sign>0,cp.isfinite(acc),12),(path>=.55,cp.isfinite(path),10),(atr>=1.05,cp.isfinite(atr),10),(htf*sign>=0,cp.isfinite(htf),12),(o5>0,cp.isfinite(o5),10),(o10>0,cp.isfinite(o10),7),(pa>0,cp.isfinite(pa),10),(va>=1.1,cp.isfinite(va),5),(iv>=-.02,cp.isfinite(iv),1),(dc*sign>=0,cp.isfinite(dc),1)]: add(cond,av,wt)
 mom=cp.where(weight>0,100*total/weight,cp.nan)
 et=cp.zeros(n,dtype=cp.float32); ew=cp.zeros(n,dtype=cp.float32)
 def eadd(cond,av,wt):
  nonlocal et,ew; et+=cp.where(av,cond.astype(cp.float32)*wt,0); ew+=cp.where(av,wt,0)
 ext=a('ema20_distance_atr'); run=a('direction_run_5m'); ds=a('direction_sign_5m'); body=a('body_to_range_5m'); cl=a('close_location_5m')
 rejection=cl<.55 if p=='ce' else cl>.45
 for cond,av,wt in [(ext*sign>1.5,cp.isfinite(ext),25),((run>=5)&(ds==sign),cp.isfinite(run)&cp.isfinite(ds),20),(acc*sign<0,cp.isfinite(acc),20),(body<.35,cp.isfinite(body),10),(rejection,cp.isfinite(cl),10),(iv>.10,cp.isfinite(iv),15)]: eadd(cond,av,wt)
 exh=cp.where(ew>0,100*et/ew,cp.nan); m=cp.asnumpy(mom).astype(np.float32); e=cp.asnumpy(exh).astype(np.float32); release(); return m,e

def select_candidates(day,dmin,dmax,step):
 dt(day); clean(day); x=day[day.option_type.isin(['CE','PE']) & day.close.gt(0) & day.strike.gt(0)].copy()
 if 'dte' in x:
  d=pd.to_numeric(x.dte,errors='coerce'); x=x[d.isna()|((d>=0)&(d<=10))]
 if x.empty: return pd.DataFrame()
 x['_exp']=x.expiry_date.dt.normalize(); x=x[x._exp==x.groupby('timestamp')['_exp'].transform('min')].copy(); x['candidate_selection_score']=rank_cuda(x,dmin,dmax,step); idx=x.groupby(['timestamp','option_type'])['candidate_selection_score'].idxmax(); s=x.loc[idx].sort_values(['timestamp','option_type'])
 common=[c for c in COMMON if c in s]; opts=[c for c in OPTION if c in s]; rows=[]
 for ts,g in s.groupby('timestamp',sort=False):
  r={'timestamp':ts}; first=g.iloc[0]
  for c in common:
   if c!='timestamp': r[c]=first.get(c,np.nan)
  for typ,p in [('CE','ce'),('PE','pe')]:
   z=g[g.option_type==typ]
   if z.empty: continue
   rr=z.iloc[0]
   for c in opts: r[f'{p}_{c}']=rr.get(c,np.nan)
   r[f'{p}_selection_score']=rr.candidate_selection_score
  rows.append(r)
 w=pd.DataFrame(rows)
 if w.empty:return w
 for p in ('ce','pe'):
  m,e=side_scores(w,p); w[f'{p}_momentum_score']=m; w[f'{p}_exhaustion_score']=e; sym=w.get(f'{p}_groww_symbol',pd.Series(index=w.index,dtype=object)); w[f'{p}_candidate_available']=sym.notna().astype('int8'); w[f'{p}_momentum_eligible']=(w[f'{p}_candidate_available'].eq(1)&w[f'{p}_momentum_score'].ge(65)&(w[f'{p}_exhaustion_score'].isna()|w[f'{p}_exhaustion_score'].le(55))).astype('int8')
 return w.replace([np.inf,-np.inf],np.nan)

def labels_cuda(entry,highs,lows,valid):
 e=cp.asarray(entry,dtype=cp.float32)
 h=cp.asarray(highs,dtype=cp.float32)
 l=cp.asarray(lows,dtype=cp.float32)
 v=cp.asarray(valid,dtype=cp.bool_)
 n,hmax=h.shape
 ev=cp.isfinite(e)&v[:,0]
 out={
  'entry_price':cp.asnumpy(cp.where(ev,e,cp.nan)).astype(np.float32),
  'entry_valid':cp.asnumpy(ev).astype(np.int8)
 }

 # MFE/MAE with explicit empty-horizon handling.
 # Avoid cp.nanmax/cp.nanmin because an all-NaN row emits RuntimeWarning.
 for mins,bars in HORIZONS.items():
  bars=min(int(bars),hmax)
  hs=h[:,:bars]
  ls=l[:,:bars]
  vs=v[:,:bars]

  vh=vs&cp.isfinite(hs)
  vl=vs&cp.isfinite(ls)

  has_h=cp.any(vh,axis=1)
  has_l=cp.any(vl,axis=1)

  safe_h=cp.where(vh,hs,-cp.inf)
  safe_l=cp.where(vl,ls, cp.inf)

  mx=cp.max(safe_h,axis=1)
  mn=cp.min(safe_l,axis=1)

  mx=cp.where(has_h,mx,cp.nan)
  mn=cp.where(has_l,mn,cp.nan)

  mfe=cp.where(ev&has_h&cp.isfinite(mx),mx-e,cp.nan)
  mae=cp.where(ev&has_l&cp.isfinite(mn),mn-e,cp.nan)

  out[f'mfe_{mins}m']=cp.asnumpy(mfe).astype(np.float32)
  out[f'mae_{mins}m']=cp.asnumpy(mae).astype(np.float32)

 # First stop touch across the configured future horizon.
 fs=cp.full(n,-1,dtype=cp.int16)
 st=v&cp.isfinite(l)&ev[:,None]&(l<=e[:,None]-STOP)
 for k in range(hmax):
  fs=cp.where((fs<0)&st[:,k],k+1,fs)

 out['time_to_stop_bars']=cp.asnumpy(
  cp.where(fs>=0,fs.astype(cp.float32),cp.nan)
 ).astype(np.float32)

 # First target touch. If target and stop occur in the same 5m bar,
 # stop wins conservatively because target must be strictly earlier.
 for target in TARGETS:
  ft=cp.full(n,-1,dtype=cp.int16)
  touch=v&cp.isfinite(h)&ev[:,None]&(h>=e[:,None]+target)

  for k in range(hmax):
   ft=cp.where((ft<0)&touch[:,k],k+1,ft)

  hit=cp.where(
   ft>=0,
   cp.where((fs<0)|(ft<fs),1.0,0.0),
   cp.where(fs>=0,0.0,cp.nan)
  )

  t=int(target)
  out[f'hit_{t}_before_stop']=cp.asnumpy(hit).astype(np.float32)
  out[f'time_to_{t}_bars']=cp.asnumpy(
   cp.where(ft>=0,ft.astype(cp.float32),cp.nan)
  ).astype(np.float32)

 release()
 return out

def side_labels(day,cands,p):
 symcol=f'{p}_groww_symbol'; base=pd.DataFrame({'timestamp':cands.timestamp}); base[symcol]=cands.get(symcol,pd.Series(index=cands.index,dtype=object)); src=day[['timestamp','groww_symbol','open','high','low']].copy(); dt(src); src=src.dropna(subset=['timestamp','groww_symbol']).drop_duplicates(['groww_symbol','timestamp'],keep='last').sort_values(['groww_symbol','timestamp']); maps={}
 for sym,g in src.groupby('groww_symbol',sort=False): maps[str(sym)]={np.datetime64(r.timestamp,'ns'):(float(r.open) if pd.notna(r.open) else np.nan,float(r.high) if pd.notna(r.high) else np.nan,float(r.low) if pd.notna(r.low) else np.nan) for r in g.itertuples(index=False)}
 n=len(base); mh=max(HORIZONS.values()); entry=np.full(n,np.nan,np.float32); hi=np.full((n,mh),np.nan,np.float32); lo=np.full((n,mh),np.nan,np.float32); valid=np.zeros((n,mh),bool)
 for i,r in base.iterrows():
  sym=r[symcol]
  if pd.isna(sym) or str(sym) not in maps: continue
  mp=maps[str(sym)]; ts=pd.Timestamp(r.timestamp)
  for k in range(1,mh+1):
   rec=mp.get(np.datetime64(ts+pd.Timedelta(minutes=5*k),'ns'))
   if rec is None: continue
   op,h,l=rec
   if k==1: entry[i]=op
   if np.isfinite(h) and np.isfinite(l): hi[i,k-1]=h; lo[i,k-1]=l; valid[i,k-1]=True
 lab=labels_cuda(entry,hi,lo,valid)
 for name,a in lab.items(): base[f'label_{p}_{name}']=a
 return base

def process_day(path,out,dmin,dmax,step):
 day=pd.read_parquet(path); dt(day); clean(day)
 if day.empty: atomic_parquet(day,out); return
 c=select_candidates(day,dmin,dmax,step)
 if c.empty: atomic_parquet(c,out); return
 for p in ('ce','pe'):
  l=side_labels(day,c,p); c=c.merge(l,on=['timestamp',f'{p}_groww_symbol'],how='left')
 atomic_parquet(c,out); del day,c; release()

def combine(globstr,out,threads,mem,temp):
 c=db(threads,mem,temp); c.execute(f"COPY (SELECT * FROM read_parquet('{globstr.replace(chr(39),chr(39)*2)}') ORDER BY timestamp) TO '{q(out)}' (FORMAT PARQUET, COMPRESSION ZSTD)"); c.close()
def report(src,out,threads,mem,temp):
 c=db(threads,mem,temp); s=src.replace("'","''"); r=c.execute(f"""SELECT COUNT(*) AS row_count, AVG(CAST(ce_candidate_available AS DOUBLE)) ce_available_fraction, AVG(CAST(pe_candidate_available AS DOUBLE)) pe_available_fraction, AVG(CAST(ce_momentum_eligible AS DOUBLE)) ce_eligible_fraction, AVG(CAST(pe_momentum_eligible AS DOUBLE)) pe_eligible_fraction, AVG(CAST(label_ce_entry_valid AS DOUBLE)) ce_entry_valid_fraction, AVG(CAST(label_pe_entry_valid AS DOUBLE)) pe_entry_valid_fraction, AVG(CAST(chain_oi_available AS DOUBLE)) chain_oi_available_fraction FROM read_parquet('{s}')""").df().iloc[0].to_dict(); c.close(); r={k:(int(v) if k=='row_count' else float(v) if pd.notna(v) else None) for k,v in r.items()}; atomic_json(r,out/'candidate_quality_report.json'); return r
def manifest(sample,out):
 sch=pq.ParquetFile(sample).schema_arrow; ids={'timestamp','bar_start','groww_symbol','expiry_date','option_type','strike'}; m={}
 for f in sch:
  if f.name.startswith('label_'): role,allow='diagnostic_label',False
  elif f.name in ids or f.name.endswith('_groww_symbol'): role,allow='identifier',False
  else: role,allow='feature',True
  m[f.name]={'dtype':str(f.type),'role':role,'allowed_in_model_observation':allow}
 atomic_json(m,out/'candidate_feature_manifest.json')

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',default='data/banknifty_5m'); ap.add_argument('--option-file',default=None); ap.add_argument('--output-dir',default='data/banknifty_5m_candidates'); ap.add_argument('--delta-min',type=float,default=.45); ap.add_argument('--delta-max',type=float,default=.60); ap.add_argument('--strike-step',type=float,default=100.0); ap.add_argument('--duckdb-threads',type=int,default=2); ap.add_argument('--duckdb-memory-gb',type=float,default=1.5); ap.add_argument('--no-combine',action='store_true'); ap.add_argument('--fresh',action='store_true'); a=ap.parse_args(); inp=Path(a.input_dir); out=Path(a.output_dir)
 if a.fresh and out.exists(): shutil.rmtree(out)
 parts=out/'candidate_parts'; extracts=out/'_daily_extract'; temp=out/'duckdb_temp'; parts.mkdir(parents=True,exist_ok=True); extracts.mkdir(parents=True,exist_ok=True); info=cuda_info(); print(f"CUDA: {info['name']} | CC={info['cc']} | free={info['free_gb']:.2f}/{info['total_gb']:.2f}GB"); print(f"DuckDB cap={a.duckdb_memory_gb:.2f}GB threads={a.duckdb_threads}"); atomic_json(info,out/'cuda_info.json'); mode,src=source_mode(inp,a.option_file); jobs=[]
 if mode=='parts':
  for p in src:
   txt=p.stem.replace('day_',''); day=pd.to_datetime(txt,format='%Y%m%d',errors='coerce')
   if pd.isna(day): day=pd.to_datetime(pd.read_parquet(p,columns=['timestamp']).iloc[0,0]).normalize()
   jobs.append((day,p,False))
 else:
  for day in days_single(src,a.duckdb_threads,a.duckdb_memory_gb,temp): jobs.append((day,extracts/f'day_{day:%Y%m%d}.parquet',True))
 print(f'Trading days: {len(jobs)}')
 for i,(day,p,need) in enumerate(jobs,1):
  tag=f'{day:%Y%m%d}'; op=parts/f'day_{tag}.parquet'
  if op.exists(): print(f'[{i:04d}/{len(jobs):04d}] {tag} cached'); continue
  print(f'[{i:04d}/{len(jobs):04d}] {tag}',flush=True)
  if need: extract_day(src,day,p,a.duckdb_threads,a.duckdb_memory_gb,temp)
  process_day(p,op,a.delta_min,a.delta_max,a.strike_step)
  if need:
   try:p.unlink()
   except FileNotFoundError:pass
  release()
 pf=sorted(parts.glob('day_*.parquet'))
 if not pf: raise RuntimeError('No candidate parts')
 final=out/'candidate_dataset.parquet'
 if a.no_combine: src_report=str(parts/'day_*.parquet'); sample=pf[0]
 else:
  if not final.exists(): combine(str(parts/'day_*.parquet'),final,a.duckdb_threads,a.duckdb_memory_gb,temp)
  src_report=str(final); sample=final
 r=report(src_report,out,a.duckdb_threads,a.duckdb_memory_gb,temp); manifest(sample,out); print('\n5-MINUTE CANDIDATE DATASET COMPLETE'); print(json.dumps(r,indent=2)); print('Dataset:', parts if a.no_combine else final)
if __name__=='__main__': main()
