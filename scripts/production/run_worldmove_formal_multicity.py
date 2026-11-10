#!/usr/bin/env python3
"""Exposure calculations and monthly/period aggregation for the frozen 33-city frame."""
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
from run_worldmove_five_city_pilot import (ROOT, load_worldmove, infer_anchor, restricted_anchor,
    exposure_arrays, summarize_exposure, software_qa, sha256_file, read_csv, qc_row,
    nearest_axis_indices, validate_netcdf_inputs)
from run_worldmove_temporal_stability import grouped_summary
from run_worldmove_temporal_v2 import MONTHS
P=ROOT/'data/processed'; T=ROOT/'outputs/tables'; AUD=P/'worldmove_frame_audit'
FRAME=P/'worldmove_formal_city_frame.json'
CACHE=P/'worldmove_formal_o3_cache'
REF=['Beijing','Shanghai','Nanchang','Chengdu','Zhengzhou']

def now(): return datetime.now(timezone.utc).isoformat()
def dump(path,obj): path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))
def table(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)



def load_cities(f,ids):
    data={}; counts=[]
    for e in f['eligible_cities']:
        c=e['city_name']; path=ROOT/e['npz_path']; assert sha256_file(path)==e['npz_sha256']
        tr,xy,meta=load_worldmove(path)
        with np.load(path,allow_pickle=True) as z: assert np.issubdtype(z['traj'].dtype,np.integer)
        assert len(tr)==e['raw_schedule_count'] and len(xy)==e['grid_cells']
        assert np.isfinite(xy).all() and xy.shape==(len(xy),2)
        assert np.prod(meta['pop_shape'])==len(xy) and meta['poi_shape']==meta['pop_shape']+[34]
        # Recompute full-pool main anchors; no alternative-anchor execution.
        full=infer_anchor(tr,np.arange(12)); assert full.valid.sum()==e['main_anchor_valid_count']
        sample=tr[ids[c]]; anchor=infer_anchor(sample,np.arange(12))
        high=restricted_anchor(anchor,anchor.valid&(anchor.dominance>=.75))
        data[c]={'traj':sample,'anchor':anchor,'high':high,'xy':xy,'full_traj':tr,'full_anchor':full,'entry':e}
        counts.append({'city':c,'release_raw_n':len(tr),'sampled_raw_n':5000,
                       'main_anchor_valid_n':int(anchor.valid.sum()),'high_confidence_valid_n':int(high.valid.sum())})
        print('STRUCTURE_AND_ANCHOR_PASS '+c,flush=True)
    return data,counts

def calculate(c,dd,dates,lookup):
    daily=[]; highrows=[]; diurnal=[]; qc=[]
    sums=defaultdict(lambda:np.zeros(24)); ns=defaultdict(int)
    tr,a,h=dd['traj'],dd['anchor'],dd['high']
    for day in dates:
        o=lookup(day); main=exposure_arrays(tr,a,o); high=exposure_arrays(tr,h,o)
        sn,se,un,ue=software_qa(tr,a,main); assert max(se,ue)<=1e-12
        assert np.all(high.complete<=main.complete) and high.complete.any()
        qc.append(qc_row('software_qa','stationary_and_uniform','PASS',city=c,date=day,value=max(se,ue),details=f'stationary_n={sn}; uniform_n={un}'))
        pr=summarize_exposure(c,day,'continuous_may_september','primary','main_0000_0600','unique_mode_dominance_ge_0.5',5000,a,main)
        hr=summarize_exposure(c,day,'continuous_may_september','high_confidence','main_0000_0600','unique_mode_dominance_ge_0.75',5000,h,high)
        common=high.complete
        hr.update(common_valid_schedule_count=int(common.sum()),primary_mean_delta_on_common_ug_m3=float(main.delta[common].mean()),
                  high_confidence_minus_primary_mean_delta_on_common_ug_m3=float((high.delta[common]-main.delta[common]).mean()),
                  primary_mean_abs_delta_on_common_ug_m3=float(np.abs(main.delta[common]).mean()),
                  high_confidence_minus_primary_mean_abs_delta_on_common_ug_m3=float(np.abs(high.delta[common]).mean()-np.abs(main.delta[common]).mean()))
        daily.append(pr); highrows.append(hr)
        for variant,result in [('primary',main),('high_confidence',high)]:
            values=result.diurnal_delta[result.complete]; n=len(values)
            for hour in range(24):
                diurnal.append({'aggregation':'daily','city':c,'month':day[:7],'date':day,'analysis_variant':variant,
                  'hour_gmt_plus_8':hour,'mean_Dt_minus_Rt_ug_m3':float(values[:,hour].mean()),'valid_pair_count':n})
            for period in (day[:7],'May_September'):
                sums[variant,period]+=values.sum(axis=0); ns[variant,period]+=n
    for (variant,period),s in sums.items():
        for hour in range(24):
            diurnal.append({'aggregation':'monthly' if period in MONTHS else 'may_september_pooled','city':c,
              'month':period,'date':'','analysis_variant':variant,'hour_gmt_plus_8':hour,
              'mean_Dt_minus_Rt_ug_m3':float(s[hour]/ns[variant,period]),'valid_pair_count':ns[variant,period]})
    return daily,highrows,diurnal,qc

def aggregates(daily,cities,variant):
    monthly=[grouped_summary(daily,c,m,variant) for c in cities for m in MONTHS]
    pooled=[grouped_summary(daily,c,None,variant) for c in cities]
    for r in pooled:
        r['month']='May_September'; rows=[x for x in monthly if x['city']==r['city']]
        for m,x in zip(MONTHS,rows):
            k=datetime.fromisoformat(m+'-01').strftime('%B').lower()
            for metric in ('delta','abs_delta'): r[f'{k}_mean_{metric}_ug_m3']=x[f'date_equal_mean_{metric}_ug_m3']
        for metric in ('delta','abs_delta'): r[f'month_mean_{metric}_range_ug_m3']=float(np.ptp([x[f'date_equal_mean_{metric}_ug_m3'] for x in rows]))
    return monthly,pooled

def compare(old,new,keys,label,tol=1e-10):
    a={tuple(str(r[k]) for k in keys):r for r in old}; b={tuple(str(r[k]) for k in keys):r for r in new}
    assert set(a)==set(b),(label,len(a),len(b)); error=0.; n=0
    for k,r in a.items():
        for field,v in r.items():
            if field not in b[k]: raise ValueError((label,field,'missing'))
            try: x=float(v); y=float(b[k][field])
            except (ValueError,TypeError):
                assert v==str(b[k][field]),(label,k,field,v,b[k][field]); continue
            assert np.isfinite(x) and np.isfinite(y)
            error=max(error,abs(x-y)); n+=1
    assert error<=tol,(label,error)
    return qc_row('reference_regression',label,'PASS',value=error,details=f'numeric_comparisons={n}; tolerance={tol}')

def extract_day(item):
    day,paths,mapping,expected_frame=item
    dest=CACHE/(day+'.npz')
    if dest.exists():
        with np.load(dest) as z: assert str(z['frame_sha256'])==expected_frame
        return day,'cached'
    arrays={c:np.empty((len(m['ix']),24),np.float32) for c,m in mapping.items()}
    # One opening per hour for all cities; same decode and float32 cast as inherited engine.
    for hour,path in enumerate(paths):
        with Dataset(path) as ds:
            v=ds['O3']
            for c,m in mapping.items():
                y,x=np.array(m['iy']),np.array(m['ix']); y0,y1=y.min(),y.max(); x0,x1=x.min(),x.max()
                crop=v[y0:y1+1,x0:x1+1]
                crop=np.asarray(crop.filled(np.nan) if np.ma.isMaskedArray(crop) else crop,dtype=np.float32)
                arrays[c][:,hour]=crop[y-y0,x-x0]
    with np.load(AUD/'daily_masks'/f'{day}.npz') as masks:
        for c,m in mapping.items(): assert np.array_equal(~np.isfinite(arrays[c]),masks[m['released_key']].T),(c,day,'mask')
    arrays['frame_sha256']=np.array(expected_frame)
    tmp=dest.with_suffix('.tmp.npz'); np.savez_compressed(tmp,**arrays); tmp.rename(dest)
    return day,'raw_decoded_and_mask_verified'


