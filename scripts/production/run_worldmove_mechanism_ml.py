#!/usr/bin/env python3
"""Frozen supplement-only XGBoost with 33 city-grouped held-out folds."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from audit_worldmove_mechanism_results import predictor_audit, sha

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'data/processed';T=ROOT/'outputs/tables'
MODEL_MANIFEST=P/'worldmove_mechanism_model_manifest.json'
CV_MANIFEST=P/'worldmove_mechanism_ml_cv_manifest.json'
ML_MANIFEST=P/'worldmove_mechanism_ml_manifest.json'
OUT={name:T/f'worldmove_mechanism_ml_{name}.csv' for name in ['oof_predictions','city_scores','summary']}


def score(y,p):
    return dict(mae=float(np.mean(np.abs(p-y))),rmse=float(np.sqrt(np.mean((p-y)**2))),
                bias=float(np.mean(p-y)),within_rmse=float(np.sqrt(np.mean(((p-p.mean())-(y-y.mean()))**2))))


def main():
    if ML_MANIFEST.exists() or any(p.exists() for p in OUT.values()):
        raise FileExistsError('ML artifacts already exist; no overwrite, tuning or rerun')
    predictor_audit()
    cfg=json.loads((P/'worldmove_mechanism_design_config.json').read_text())
    assert cfg['ML_ROLE']=='SUPPLEMENT_ONLY' and cfg['RF_role']=='NONE'
    model_manifest=json.loads(MODEL_MANIFEST.read_text())
    fold_manifest=json.loads(CV_MANIFEST.read_text())
    assert model_manifest['status']=='FORMAL_LOW_DIMENSIONAL_MODELS_COMPLETE'
    assert model_manifest['cv_manifest_sha256']==sha(CV_MANIFEST)
    assert model_manifest['predictor_manifest_sha256']==sha(P/'worldmove_mechanism_predictor_manifest.json')
    for path,h in model_manifest['output_sha256'].items():assert sha(ROOT/path)==h,path
    for path,h in model_manifest['outcome_file_sha256'].items():assert sha(ROOT/path)==h,path
    features=cfg['xgboost']['predictors']
    assert features==fold_manifest['feature_names']==['time_away_fraction','mean_away_distance_km',
                                                      'pooled_spatial_iqr','within_spatial_iqr','within_alignment_excess']
    city=pd.read_csv(T/'worldmove_mechanism_predictors_city.csv')
    day=pd.read_csv(T/'worldmove_mechanism_predictors_city_date.csv')
    y=pd.read_csv(T/'worldmove_formal_city_date.csv',usecols=['city','date','mean_abs_delta_ug_m3'])
    c=city[city.variant=='primary'][['city','T','E']].rename(columns={'T':features[0],'E':features[1]})
    d=day[day.variant=='primary'][['city','date','Hbar','Hw','Aw']].rename(columns={
        'Hbar':features[2],'Hw':features[3],'Aw':features[4]})
    data=d.merge(c,on='city',validate='many_to_one').merge(y,on=['city','date'],validate='one_to_one')
    assert len(data)==5049 and set(data.city)==set(cfg['formal_frame']['cities'])
    assert not data[features+['mean_abs_delta_ug_m3']].isna().any().any()
    X=data[features].to_numpy(dtype=np.float64);Y=data.mean_abs_delta_ug_m3.to_numpy(dtype=np.float64)
    groups=data.city.to_numpy();logo=LeaveOneGroupOut()
    pred={key:np.full(len(data),np.nan,dtype=np.float64) for key in ('xgboost','linear','null')}
    train_means={}
    fold_order=[]
    for train,test in logo.split(X,Y,groups):
        city_name=str(np.unique(groups[test])[0]);fold_order.append(city_name)
        assert len(test)==153 and len(np.unique(groups[train]))==32
        assert not set(groups[train])&set(groups[test])
        train_mean=float(np.mean(Y[train]));train_means[city_name]=train_mean
        scaler=StandardScaler().fit(X[train])
        linear=LinearRegression().fit(scaler.transform(X[train]),Y[train])
        pred['linear'][test]=linear.predict(scaler.transform(X[test]))
        pred['null'][test]=train_mean
        params=dict(cfg['xgboost']['parameters'])
        model=XGBRegressor(**params,base_score=train_mean)
        model.fit(X[train],Y[train])
        pred['xgboost'][test]=model.predict(X[test])
        print('ML_GROUP_FOLD',len(fold_order),'/',33,city_name,flush=True)
    assert set(fold_order)==set(cfg['formal_frame']['cities'])
    assert all(np.isfinite(v).all() for v in pred.values())
    oof=data[['city','date']].copy()
    oof['observed_Y_ug_m3']=Y
    for name,values in pred.items():
        oof[name+'_prediction']=values
        oof[name+'_error']=values-Y
    city_scores=[]
    for city_name in cfg['formal_frame']['cities']:
        index=(groups==city_name)
        out=dict(city=city_name,n_days=int(index.sum()),train_mean_Y=train_means[city_name])
        for name,values in pred.items():
            metrics=score(Y[index],values[index])
            out.update({name+'_'+key:value for key,value in metrics.items()})
        out['mae_gain_linear_minus_xgb']=out['linear_mae']-out['xgboost_mae']
        out['xgb_negative_prediction_n']=int(np.sum(pred['xgboost'][index]<0))
        city_scores.append(out)
    scores=pd.DataFrame(city_scores)
    summary=[]
    for name,values in pred.items():
        by_city=scores[[name+'_mae',name+'_rmse',name+'_within_rmse']]
        observed_anomaly=np.empty_like(Y)
        predicted_anomaly=np.empty_like(Y)
        for city_name in cfg['formal_frame']['cities']:
            index=(groups==city_name)
            observed_anomaly[index]=Y[index]-Y[index].mean()
            predicted_anomaly[index]=values[index]-values[index].mean()
        summary.append(dict(model=name,macro_mae=float(by_city[name+'_mae'].mean()),
                            macro_rmse=float(by_city[name+'_rmse'].mean()),
                            macro_within_rmse=float(by_city[name+'_within_rmse'].mean()),
                            pooled_within_rmse=float(np.sqrt(np.mean((predicted_anomaly-observed_anomaly)**2))),
                            oof_r2=float(r2_score(Y,values)),
                            demeaned_within_r2=float(r2_score(observed_anomaly,predicted_anomaly)),
                            negative_prediction_count=int(np.sum(values<0)),n_city=33,n_city_date=5049))
    performance=pd.DataFrame(summary)
    perf=performance.set_index('model')
    x=perf.loc['xgboost'];l=perf.loc['linear'];n=perf.loc['null']
    relative_gain=float((l.macro_mae-x.macro_mae)/l.macro_mae)
    improve=int((scores.mae_gain_linear_minus_xgb>0).sum())
    deletion_gains=[float(scores.loc[scores.city!=held,'mae_gain_linear_minus_xgb'].mean())
                    for held in cfg['formal_frame']['cities']]
    criteria=dict(macro_mae_reduction_at_least_5pct=bool(relative_gain>=.05),
                  at_least_22_cities_improve=bool(improve>=22),
                  macro_rmse_not_worse=bool(x.macro_rmse<=l.macro_rmse),
                  within_rmse_improves=bool(x.macro_within_rmse<l.macro_within_rmse),
                  beats_null_macro_mae=bool(x.macro_mae<n.macro_mae),
                  deletion_of_each_held_city_keeps_gain_positive=bool(min(deletion_gains)>0))
    stable=all(criteria.values())
    decision='STABLE_SUPPLEMENT_INCREMENT' if stable else 'NO_STABLE_INCREMENT'
    performance['ml_decision']=decision
    performance['relative_macro_mae_gain_vs_linear']=relative_gain
    performance['improved_city_count']=improve
    performance['min_delete_one_city_mae_gain']=min(deletion_gains)
    performance['max_delete_one_city_mae_gain']=max(deletion_gains)
    for key,val in criteria.items():performance['criterion_'+key]=val
    for name,df in [('oof_predictions',oof),('city_scores',scores),('summary',performance)]:
        with OUT[name].open('x',newline='',encoding='utf-8') as f:df.to_csv(f,index=False,float_format='%.17g')
    manifest=dict(status='SUPPLEMENT_ML_COMPLETE',completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  script_sha256=sha(Path(__file__)),config_sha256=sha(P/'worldmove_mechanism_design_config.json'),
                  predictor_manifest_sha256=sha(P/'worldmove_mechanism_predictor_manifest.json'),
                  model_manifest_sha256=sha(MODEL_MANIFEST),cv_manifest_sha256=sha(CV_MANIFEST),
                  output_sha256={str(path.relative_to(ROOT)):sha(path) for path in OUT.values()},
                  versions={'xgboost':xgboost.__version__,'sklearn':sklearn.__version__,
                            'numpy':np.__version__,'pandas':pd.__version__},
                  feature_names=features,folds=33,train_cities_per_fold=32,test_rows_per_fold=153,
                  parameters=cfg['xgboost']['parameters'],base_score='training-fold mean Y',
                  grouped_cv='LeaveOneGroupOut city',train_only_linear_scaling=True,
                  test_Y_used_for_feature_or_prediction=False,test_Y_city_mean_used_for_prediction=False,
                  retrospective_complete_window_predictors=True,
                  criteria=criteria,relative_macro_mae_gain_vs_linear=relative_gain,
                  improved_city_count=improve,min_delete_one_city_mae_gain=min(deletion_gains),
                  decision=decision,random_forest_run=False,shap_run=False)
    with ML_MANIFEST.open('x',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('SUPPLEMENT_ML_COMPLETE',json.dumps({'decision':decision,'criteria':criteria,
        'xgb_macro_mae':x.macro_mae,'linear_macro_mae':l.macro_mae,
        'xgb_macro_rmse':x.macro_rmse,'linear_macro_rmse':l.macro_rmse},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
