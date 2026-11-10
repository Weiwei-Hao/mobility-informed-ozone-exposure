#!/usr/bin/env python3
"""Explicit reproduction routes for the WorldMove local code/data release.

`subset` and `verify-results` do not fit models. No image-generation commands are included.
`models` fits the original full specifications only when explicitly invoked.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys

from release_support import DEFAULT_DATA,resolve_input

PACKAGE=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def dump(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def read_table(data,logical):
    import pandas as pd
    p=resolve_input(logical,data)
    if p is None:raise FileNotFoundError(logical)
    return pd.read_csv(p)


def verify_subset(data,report_path):
    import numpy as np
    sys.path.insert(0,str(PACKAGE/'scripts/production'))
    from run_worldmove_five_city_pilot import infer_anchor,restricted_anchor,nearest_axis_indices,exposure_arrays,summarize_exposure
    from generate_worldmove_mechanism_predictors import haversine_km
    fixture=data/'real_subset/yinchuan_20190501_inputs.npz'
    meta=json.loads(fixture.with_suffix('.json').read_text());assert sha(fixture)==meta['fixture_sha256']
    with np.load(fixture,allow_pickle=False) as a:inputs={k:a[k] for k in a.files}
    tr,xy=inputs['trajectories'],inputs['grid_lonlat']
    iy=nearest_axis_indices(inputs['ozone_lat_axis'],xy[:,1]);ix=nearest_axis_indices(inputs['ozone_lon_axis'],xy[:,0])
    cell_index={tuple(v):i for i,v in enumerate(inputs['product_cell_yx'])}
    matched=np.array([cell_index[(y,x)] for y,x in zip(iy,ix)],dtype=np.int64)
    o3=inputs['product_cell_hourly_o3'][matched]
    anchor=infer_anchor(tr,np.arange(12))
    high=restricted_anchor(anchor,anchor.valid&(anchor.dominance>=.75))
    city,date=meta['city'],meta['date'];errors={}
    for label,a in [('primary',anchor),('high_confidence',high)]:
        result=exposure_arrays(tr,a,o3)
        row=summarize_exposure(city,date,'continuous_may_september',label,'main_0000_0600',
              'unique_mode_dominance_ge_0.5' if label=='primary' else 'unique_mode_dominance_ge_0.75',5000,a,result)
        source='worldmove_formal_city_date.csv' if label=='primary' else 'worldmove_formal_high_confidence.csv'
        official=read_table(data,'outputs/tables/'+source)
        target=official[(official.city==city)&(official.date==date)].iloc[0]
        for key in ['mean_R_ug_m3','mean_D_ug_m3','mean_delta_ug_m3','mean_abs_delta_ug_m3','valid_schedule_count']:
            err=abs(float(row[key])-float(target[key]));assert err<1e-10,(label,key,err);errors[label+'/'+key]=err
    # Predictor test uses the same product support, not independent evidence.
    selected=tr[anchor.valid];ag=anchor.grid_index[anchor.valid];away=selected!=ag[:,None]
    M=away.reshape(len(selected),24,2).mean(axis=(0,2),dtype=np.float64)
    ri,si=np.nonzero(away);st,en=xy[ag[ri]],xy[selected[ri,si]]
    E=float(np.mean(haversine_km(st[:,0],st[:,1],en[:,0],en[:,1]),dtype=np.float64)) if len(ri) else 0.
    values=inputs['product_cell_hourly_o3'];values=values[np.isfinite(values).all(axis=1)].astype(np.float64)
    q=np.quantile(values,[.25,.75],axis=0,method='linear');hour_H=q[1]-q[0]
    H=float(hour_H.mean());A=float(np.dot(M,hour_H)/M.sum()-H) if M.sum()>0 else 0.
    cp=read_table(data,'outputs/tables/worldmove_mechanism_predictors_city.csv').query('variant == "primary"').set_index('city')
    dp=read_table(data,'outputs/tables/worldmove_mechanism_predictors_city_date.csv')
    dr=dp[(dp.city==city)&(dp.date==date)&(dp.variant=='primary')].iloc[0]
    for key,value,target in [('T',float(away.mean()),cp.loc[city,'T']),('E',E,cp.loc[city,'E']),('H',H,dr.H),('A',A,dr.A)]:
        err=abs(value-float(target));assert err<1e-10,(key,err);errors[key]=err
    result=dict(route='real one-city one-date input subset',city=city,date=date,sampled_schedules=len(tr),
        input_kind=meta['kind'],fixture_sha256=sha(fixture),spatial_matching='nearest native product coordinates; lower-coordinate exact ties',
        temporal_matching='48 slots, slot//2, 24 Beijing-time fields',max_abs_error=max(errors.values()),errors=errors,
        passed=True,models_fitted=False,full_33_city_153_day_run=False,
        limits='Starts from saved decoded sparse product cells, not the original national NetCDF ingestion or generation of the ozone product; fixed full-window support is inherited.')
    dump(report_path,result);print(json.dumps(result,indent=2))


def verify_results(data,report_path):
    import numpy as np
    day=read_table(data,'outputs/tables/worldmove_formal_city_date.csv')
    pooled=read_table(data,'outputs/tables/worldmove_formal_city_may_sep.csv')
    pred=read_table(data,'outputs/tables/worldmove_mechanism_predictors_city_date.csv')
    assert len(day)==5049 and len(pooled)==33 and len(pred)==10098
    assert not day.duplicated(['city','date']).any() and day.groupby('city').size().eq(153).all()
    maximum=0.
    for metric in ['delta','abs_delta']:
        means=day.groupby('city')[f'mean_{metric}_ug_m3'].mean()
        err=np.max(np.abs(means.loc[pooled.city].to_numpy()-pooled[f'date_equal_mean_{metric}_ug_m3'].to_numpy()));maximum=max(maximum,float(err))
    assert maximum<1e-10
    for raw,mean,dev in [('H','Hbar','Hw'),('A','Abar','Aw'),('U','Ubar','Uw'),('H90','H90bar','H90w'),('A90','A90bar','A90w')]:
        err=float(np.max(np.abs(pred[raw]-pred[mean]-pred[dev])));assert err<1e-10;maximum=max(maximum,err)
    # Recompute the saved city-level error summaries without training any model.
    oof=read_table(data,'outputs/tables/worldmove_mechanism_ml_oof_predictions.csv')
    scores=read_table(data,'outputs/tables/worldmove_mechanism_ml_city_scores.csv').set_index('city')
    for city,g in oof.groupby('city',sort=False):
        y=g.observed_Y_ug_m3.to_numpy()
        for model in ['xgboost','linear','null']:
            e=g[model+'_prediction'].to_numpy()-y
            for metric,value in [('mae',float(np.mean(np.abs(e)))),('rmse',float(np.sqrt(np.mean(e*e))))]:
                err=abs(value-float(scores.loc[city,model+'_'+metric]));assert err<1e-10;maximum=max(maximum,err)
    result=dict(route='packaged frozen table consistency and score reconstruction',passed=True,primary_rows=len(day),cities=33,dates=153,
                max_absolute_error=maximum,models_fitted=False,new_predictions_generated=False,
                limits='Checks stored aggregates/centering and reconstructs scores from frozen predictions; not a model-refitting validation.')
    dump(report_path,result);print(json.dumps(result,indent=2))


def derived_preflight(data):
    import pandas as pd
    rows=json.loads((PACKAGE/'config/derived_model_inputs.json').read_text());checked=[]
    for r in rows:
        p=resolve_input(r['logical_path'],data)
        if p is None:raise FileNotFoundError(r['logical_path'])
        assert sha(p)==r['sha256'],r['logical_path']
        d=pd.read_csv(p);assert len(d)==int(r['rows'])
        checked.append(dict(path=r['logical_path'],sha256=sha(p),rows=len(d)))
    return dict(status='PASS',scope='release derived-table hashes and shapes only; does not repeat raw-field/predictor-generation audit',files=checked)


def fit_models(data,work,include_ml):
    """Fit the original full model/sensitivity specifications only when explicitly selected."""
    result=derived_preflight(data)
    if work.exists():raise FileExistsError('Choose a new output directory.')
    if any(work==p or p in work.parents for p in [data,PACKAGE]):raise ValueError('Output must be outside the input and code trees.')
    work.mkdir(parents=True)
    shutil.copytree(PACKAGE/'scripts/production',work/'scripts')
    p,t=work/'data/processed',work/'outputs/tables';p.mkdir(parents=True);t.mkdir(parents=True)
    for name in ['worldmove_formal_city_frame.json','worldmove_mechanism_design_config.json']:
        shutil.copyfile(PACKAGE/'config'/name,p/name)
    for row in json.loads((PACKAGE/'config/derived_model_inputs.json').read_text()):
        shutil.copyfile(resolve_input(row['logical_path'],data),t/Path(row['logical_path']).name)
    # Verify supplied derived inputs without claiming a new raw-data audit.
    dump(p/'worldmove_mechanism_predictor_manifest.json',dict(status='RELEASE_DERIVED_INPUTS_VERIFIED',report=result))
    sys.path.insert(0,str(work/'scripts'))
    models=importlib.import_module('run_worldmove_mechanism_models')
    models.predictor_audit=lambda:result
    models.main()
    if include_ml:
        ml=importlib.import_module('run_worldmove_mechanism_ml');ml.predictor_audit=lambda:result;ml.main()
    dump(work/'release_model_run.json',dict(derived_input_preflight=result,include_ml=include_ml,
        preserved_numerical_code=True,scope='Refitted frozen specifications; compare against supplied frozen outputs.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=DEFAULT_DATA)
    sub=p.add_subparsers(dest='command',required=True)
    for cmd in ['subset','verify-results']:
        s=sub.add_parser(cmd);s.add_argument('--report',type=Path,required=True)
    s=sub.add_parser('check');s.add_argument('--mode',choices=['A','models'],required=True);s.add_argument('--raw-root',type=Path)
    s=sub.add_parser('models',help='Explicit full statistical refit; not a smoke test.');s.add_argument('--work-dir',type=Path,required=True);s.add_argument('--include-ml',action='store_true')
    a=p.parse_args();data=a.data_root.resolve()
    if a.command in ['subset','verify-results']:
        if a.report.exists():raise FileExistsError('Choose a new report path.')
        return (verify_subset if a.command=='subset' else verify_results)(data,a.report)
    if a.command=='models':return fit_models(data,a.work_dir.resolve(),a.include_ml)
    pipeline=importlib.import_module('release_pipeline')
    if a.mode=='models':print(json.dumps(derived_preflight(data),indent=2));return
    root=(a.raw_root or data).resolve()
    # Explicit --data-root also applies when raw inputs are elsewhere.
    pipeline.DERIVED_ROOT=data
    r=pipeline.check_inputs(a.mode,root);print(json.dumps(r,indent=2));sys.exit(0 if r['ready'] else 2)


if __name__=='__main__':main()
