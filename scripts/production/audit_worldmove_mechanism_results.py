#!/usr/bin/env python3
"""Independent numerical checks of frozen mechanism predictors."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'data/processed'
T=ROOT/'outputs/tables'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def close(a,b,label,tol=2e-10):
    x=np.asarray(a,dtype=float);y=np.asarray(b,dtype=float)
    if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise AssertionError(label+' shape/finite mismatch')
    difference=float(np.max(np.abs(x-y))) if x.size else 0
    if difference>tol:
        raise AssertionError(f'{label} mismatch {difference}')
    return difference


def predictor_audit():
    cfg=json.loads((P/'worldmove_mechanism_design_config.json').read_text())
    manifest=json.loads((P/'worldmove_mechanism_predictor_manifest.json').read_text())
    assert cfg['freeze_status']=='MECHANISM_DESIGN_FROZEN'
    assert manifest['status']=='PREDICTOR_FREEZE_COMPLETE' and manifest['qa']['all_hard_checks_passed']
    assert manifest['exact_config_sha256']==sha(P/'worldmove_mechanism_design_config.json')
    assert manifest['script_sha256']==sha(ROOT/manifest['script_path'])
    for path,h in manifest['file_sha256'].items():assert sha(ROOT/path)==h,path
    for path,h in manifest['source_hashes'].items():assert sha(ROOT/path)==h,path
    for path,h in cfg['protected_source_hashes'].items():assert sha(ROOT/path)==h,path
    for name,h in manifest['cache_sha256'].items():assert sha(P/'worldmove_formal_o3_cache'/name)==h,name
    for path,h in manifest['mask_sha256'].items():assert sha(ROOT/path)==h,path
    city=pd.read_csv(T/'worldmove_mechanism_predictors_city.csv')
    hour=pd.read_csv(T/'worldmove_mechanism_predictors_city_hour.csv')
    hourly=pd.read_csv(T/'worldmove_mechanism_predictors_city_date_hour.csv')
    daily=pd.read_csv(T/'worldmove_mechanism_predictors_city_date.csv')
    qa=pd.read_csv(T/'worldmove_mechanism_predictor_qa.csv')
    assert len(city)==66 and len(hour)==1584 and len(hourly)==121176 and len(daily)==10098
    assert len(qa)==manifest['qa']['rows']
    assert not qa.status.eq('FAIL').any() and not qa.status.eq('NON_ESTIMABLE').any()
    assert set(qa.status)=={'PASS','FLAG_GT_5'}
    assert list(city[city.variant=='primary'].city)==cfg['formal_frame']['cities']
    assert set(daily.date)==set(cfg['formal_frame']['dates'])
    for df,keys in [(city,['city','variant']),(hour,['city','variant','hour']),
                    (hourly,['city','date','hour']),(daily,['city','date','variant'])]:
        assert not df.duplicated(keys).any() and not df.isna().any().any()
    assert city['T'].between(0,1).all() and (city['E']>=0).all()
    assert (hourly[['H_iqr','H_p90_p10']]>=0).all().all()
    assert city.unique_valid_o3_cells.min()>=2
    # Reconstruct daily H/H90/U from a separate aggregation path.
    summaries=hourly.groupby(['city','date'],sort=False)[['H_iqr','H_p90_p10','footprint_mean_o3']].mean().reset_index()
    values=daily.merge(summaries,on=['city','date'],validate='many_to_one')
    e={}
    for x,y,label in [('H','H_iqr','daily_H'),('H90','H_p90_p10','daily_H90'),('U','footprint_mean_o3','daily_U')]:
        e[label]=close(values[x],values[y],label)
    m=hourly.merge(hour,on=['city','hour'],validate='many_to_many')
    m['weighted_H']=m.M*m.H_iqr
    m['weighted_H90']=m.M*m.H_p90_p10
    agg=m.groupby(['city','date','variant'],sort=False).agg(
        numerator=('weighted_H','sum'),numerator90=('weighted_H90','sum'),Msum=('M','sum'),
        H=('H_iqr','mean'),H90=('H_p90_p10','mean')).reset_index()
    check=daily.merge(agg,on=['city','date','variant'],validate='one_to_one',suffixes=('','_recomputed'))
    independent_A=np.where(check.Msum>0,check.numerator/check.Msum-check.H_recomputed,0)
    independent_A90=np.where(check.Msum>0,check.numerator90/check.Msum-check.H90_recomputed,0)
    e['A_all_variants']=close(check.A,independent_A,'A from frozen hourly M,H')
    e['A90_all_variants']=close(check.A90,independent_A90,'A90 from frozen hourly M,H90')
    t_check=hour.groupby(['city','variant'],sort=False).M.mean().reset_index().merge(city,on=['city','variant'],validate='one_to_one')
    e['T_equals_hour_mean_M']=close(t_check.M,t_check['T'],'T from hourly M')
    for x,mean,diff,label in [('H','Hbar','Hw','H'),('A','Abar','Aw','A'),('U','Ubar','Uw','U'),
                              ('H90','H90bar','H90w','H90'),('A90','A90bar','A90w','A90')]:
        e[label+'_centering']=close(daily[x]-daily[mean],daily[diff],label+' centered')
        means=daily.groupby(['city','variant'],sort=False)[x].transform('mean')
        e[label+'_city_mean']=close(means,daily[mean],label+' city mean')
    return {'status':'PASS','freeze':'PREDICTOR_FREEZE_COMPLETE','max_absolute_formula_errors':e,
            'rows':manifest['row_counts'],'vif_gt_5':qa[qa.status=='FLAG_GT_5'][['scope','variable','value']].to_dict('records'),
            'outcome_values_read':False}






