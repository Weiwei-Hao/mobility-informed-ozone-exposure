#!/usr/bin/env python3
"""Bounded EI pre-submission reproduction of the frozen primary models and LOCO.

Calls the unchanged production fitting/interval functions. The two LOCO loops
retain the production equations, ordering and held-out scoring conventions.
Does not invoke the full runner, sensitivities, XGBoost, sampling or rasters.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib, json, platform, sys
import numpy as np
import pandas as pd
import scipy, statsmodels

sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parents[1]
DEFAULT_CODE=BASE
PRODUCTION_SHA='f1a80edb0b7e5136b20f0f956ca9d996ee9683dd14fb90a278655de6a0884a0a'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def compare(actual,expected,keys,label):
    # Preserve output order in the actual CSV; joins here are only comparisons.
    assert not actual.duplicated(keys).any(),label
    assert not expected.duplicated(keys).any(),label
    cols=[c for c in actual if c not in keys and c in expected and pd.api.types.is_numeric_dtype(actual[c])]
    merged=actual.merge(expected,on=keys,suffixes=('_new','_frozen'),how='outer',validate='one_to_one',indicator=True)
    assert merged._merge.eq('both').all(),label
    errors={};valid={}
    for c in cols:
        a=merged[c+'_new'].to_numpy(dtype=float);b=merged[c+'_frozen'].to_numpy(dtype=float)
        finite=np.isfinite(a)&np.isfinite(b)
        errors[c]=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.
        valid[c]=bool(np.allclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True))
    return {'label':label,'rows':len(merged),'numeric_columns':len(cols),'maximum_absolute_discrepancy':max(errors.values(),default=0.),
            'per_column_absolute_discrepancy':errors,'per_column_pass':valid,'passed':all(valid.values())}

def execute(code,data,out):
    code,data,out=code.resolve(),data.resolve(),out.resolve()
    if out.exists():raise FileExistsError('Use a new output directory: '+str(out))
    for tree in (code,data):
        if out==tree or tree in out.parents:raise ValueError('Outputs must be outside code and input trees')
    module=code/'scripts/production/run_worldmove_mechanism_models.py'
    assert sha(module)==PRODUCTION_SHA,'Production fitting source changed; inspect rather than silently refit'
    cfg_path=code/'config/worldmove_mechanism_design_config.json'
    cfg=json.loads(cfg_path.read_text())
    cities=cfg['formal_frame']['cities'];dates=cfg['formal_frame']['dates']
    assert len(cities)==len(set(cities))==33 and len(dates)==153
    expected_dates=pd.date_range('2019-05-01','2019-09-30').strftime('%Y-%m-%d').tolist()
    assert dates==expected_dates
    inputs=[]
    for row in json.loads((code/'config/derived_model_inputs.json').read_text()):
        p=data/row['stored_path']
        assert sha(p)==row['sha256'],row['logical_path']
        assert len(pd.read_csv(p))==int(row['rows'])
        inputs.append({'path':row['stored_path'],'sha256':sha(p),'rows':int(row['rows'])})
    sys.path.insert(0,str(code/'scripts/production'))
    m=importlib.import_module('run_worldmove_mechanism_models')
    m.T=data/'tables'  # I/O only; source and numerical functions unchanged.
    daily,pooled,_,_=m.read_outcomes(cfg)
    city=pd.read_csv(m.T/'worldmove_mechanism_predictors_city.csv')
    day=pd.read_csv(m.T/'worldmove_mechanism_predictors_city_date.csv')
    primary_city=city[city.variant=='primary'].merge(pooled,on='city',validate='one_to_one')
    primary_day=day[day.variant=='primary'].merge(daily[['city','date','mean_abs_delta_ug_m3']],on=['city','date'],validate='one_to_one')
    assert list(primary_city.city)==cities
    assert len(primary_day)==5049 and primary_day.groupby('city',sort=False).size().eq(153).all()
    yb='date_equal_mean_abs_delta_ug_m3';yw='mean_abs_delta_ug_m3'
    core_b=['T','E','Hbar'];core_w=['Hw','Aw']
    br,bfit=m.between_rows(primary_city,core_b,yb,'B_full',True)
    bhr,bhfit=m.between_rows(primary_city,['Hbar'],yb,'B_Hbar_only')
    wr,wfit=m.within_rows(primary_day,core_w,yw,'W_full')
    whr,whfit=m.within_rows(primary_day,['Hw'],yw,'W_H_only')
    between_primary=pd.DataFrame(br+bhr);within_primary=pd.DataFrame(wr+whr)
    # Original between LOCO block, source lines 171-183; no new model/fold.
    between_loco=[]
    for city_name in cities:
        train=primary_city[primary_city.city!=city_name];test=primary_city[primary_city.city==city_name]
        fit=m.fit_between(train,core_b,yb,'B3_LOCO_'+city_name)
        prediction=float(np.r_[1.,test[core_b].to_numpy()[0]]@fit['beta'])
        observed=float(test[yb].iloc[0])
        for k,term in enumerate(['intercept']+core_b):
            between_loco.append(dict(held_out_city=city_name,term=term,estimate=fit['beta'][k],
                full_model_estimate=bfit['beta'][k],coefficient_change=fit['beta'][k]-bfit['beta'][k],
                held_out_prediction=prediction,held_out_observed=observed,held_out_error=prediction-observed,
                held_out_abs_error=abs(prediction-observed),training_cities=32))
    # Original within LOCO block, source lines 206-230. Test Y enters scoring only.
    within_loco=[];within_loco_pred=[]
    for held in cities:
        train=primary_day[primary_day.city!=held]
        test=primary_day[primary_day.city==held].copy()
        train_y=(train[yw]-train.groupby('city',sort=False)[yw].transform('mean')).to_numpy()
        train_x=train[core_w].to_numpy()
        beta=np.linalg.lstsq(train_x,train_y,rcond=None)[0]
        pred=test[core_w].to_numpy()@beta
        beta_reduced=np.linalg.lstsq(train_x[:,:1],train_y,rcond=None)[0]
        pred_reduced=test[['Hw']].to_numpy()@beta_reduced
        observed=test[yw].to_numpy();actual_anomaly=observed-observed.mean()
        error=pred-actual_anomaly;reduced_error=pred_reduced-actual_anomaly
        for k,term in enumerate(core_w):
            within_loco.append(dict(held_out_city=held,term=term,estimate=beta[k],
                full_model_estimate=wfit['beta'][k],coefficient_change=beta[k]-wfit['beta'][k],
                held_city_anomaly_rmse=np.sqrt(np.mean(error**2)),held_city_H_only_rmse=np.sqrt(np.mean(reduced_error**2))))
        for date,prediction,reduced_prediction,actual in zip(test.date,pred,pred_reduced,actual_anomaly):
            within_loco_pred.append(dict(city=held,date=date,predicted_anomaly=prediction,
                H_only_predicted_anomaly=reduced_prediction,observed_anomaly_for_scoring_only=actual,
                error=prediction-actual,H_only_error=reduced_prediction-actual))
    outputs={'between_primary':between_primary,'within_primary':within_primary,
             'between_loco':pd.DataFrame(between_loco),'within_loco':pd.DataFrame(within_loco),
             'within_loco_predictions':pd.DataFrame(within_loco_pred)}
    comparisons=[];references=[]
    for name,df in outputs.items():
        p=m.T/f'worldmove_mechanism_{name}.csv'
        reference=pd.read_csv(p,float_precision='round_trip')
        if name=='between_primary':reference=reference[reference.model.isin(['B_full','B_Hbar_only'])]
        keys=['model','term'] if name.endswith('primary') else (['city','date'] if name.endswith('predictions') else ['held_out_city','term'])
        comparisons.append(compare(df,reference,keys,name))
        references.append({'path':str(p.relative_to(data)),'sha256':sha(p)})
    wl=outputs['within_loco'].drop_duplicates('held_out_city')
    bl=outputs['between_loco'].drop_duplicates('held_out_city')
    scores={'between_r2':bfit['r2'],'between_adjusted_r2':bfit['adjusted_r2'],
        'between_Hbar_only_r2':bhfit['r2'],'between_Hbar_only_adjusted_r2':bhfit['adjusted_r2'],
        'within_H_only_r2':whfit['within_r2'],'within_H_A_r2':wfit['within_r2'],
        'within_r2_increment':wfit['within_r2']-whfit['within_r2'],
        'within_loco_H_only_macro_rmse':float(wl.held_city_H_only_rmse.mean()),
        'within_loco_H_A_macro_rmse':float(wl.held_city_anomaly_rmse.mean()),
        'within_loco_improved_cities':int((wl.held_city_anomaly_rmse<wl.held_city_H_only_rmse).sum()),
        'within_loco_total_cities':len(wl),'between_loco_mae':float(bl.held_out_abs_error.mean())}
    rounded={'within_H_only_r2':'0.445','within_H_A_r2':'0.576',
             'within_loco_H_only_macro_rmse':'0.426','within_loco_H_A_macro_rmse':'0.371'}
    rounding_checks={k:{'reproduced':format(scores[k],'.3f'),'manuscript':v,
                        'passed':format(scores[k],'.3f')==v} for k,v in rounded.items()}
    settings={'between':'OLS, intercept + T/E/Hbar; equal city weight; HC3; model-specific t (29 full, 31 Hbar-only)',
        'within':'City demeaned outcome and features; no explicit intercept; cov_nw_groupsum nlags=7, use_correction=hac; fixed t32',
        'within_covariance_factors':{'W_full':5049/(5049-2),'W_H_only':5049/(5049-1)},
        'within_r2_denominator':'sum of squares of city-demeaned Y; no adjusted within R2 reported or newly defined',
        'predictor_sd':'ddof=1; between 33 city rows, within pooled city-demeaned rows',
        'within_loco':'Original np.linalg.lstsq rcond=None; full-window predictor centering; held-out Y mean used only to score anomalies',
        'macro_rmse':'Compute each held-city RMSE over 153 dates, then arithmetic mean over 33 cities (not pooled RMSE)',
        'data_order':'Original stored predictor row order and production merge order; no new sorting or sampling'}
    all_pass=all(c['passed'] for c in comparisons) and all(c['passed'] for c in rounding_checks.values()) and scores['within_loco_improved_cities']==32
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Only frozen between/within primary and reduced models, and original 33-city LOCO',
        'passed':all_pass,'production_model_source_sha256':sha(module),'config_sha256':sha(cfg_path),'verification_script_sha256':sha(Path(__file__)),
        'inputs':inputs,'references':references,'settings':settings,'software_versions':{'python':platform.python_version(),
             'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'statsmodels':statsmodels.__version__},
        'fits':{'between_primary_and_Hbar_only':2,'within_primary_and_H_only':2,'between_LOCO':33,
                'within_LOCO_full_and_reduced_least_squares':66,'sensitivities':0,'XGBoost':0},
        'scores':scores,'manuscript_rounding_checks':rounding_checks,'comparisons':comparisons,
        'limits':'Finite derived-input reproduction, not full raw-input or whole-paper reproduction. No rasters, resampling, tuning, sensitivity fits or XGBoost training.'}
    out.mkdir(parents=True,exist_ok=False)
    for name,df in outputs.items():df.to_csv(out/f'reproduced_{name}.csv',index=False,float_format='%.17g')
    wl.to_csv(out/'held_city_scores.csv',index=False,float_format='%.17g')
    (out/'core_model_checks.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':all_pass,'scores':scores,'comparison_max_abs':{c['label']:c['maximum_absolute_discrepancy'] for c in comparisons}},indent=2))
    return all_pass

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-root',type=Path,default=DEFAULT_CODE)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    data=args.data_root or args.code_root/'data'
    if not execute(args.code_root,data,args.output):raise SystemExit(1)

if __name__=='__main__':main()
