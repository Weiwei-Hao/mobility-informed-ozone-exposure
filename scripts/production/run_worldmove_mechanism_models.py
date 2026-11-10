#!/usr/bin/env python3
"""Stage B: frozen low-dimensional WorldMove explanatory models and sensitivities."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr, t as student_t
from statsmodels.stats.sandwich_covariance import cov_nw_groupsum

from audit_worldmove_mechanism_results import predictor_audit, sha

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'data/processed'
T=ROOT/'outputs/tables'
MODEL_MANIFEST=P/'worldmove_mechanism_model_manifest.json'
CV_MANIFEST=P/'worldmove_mechanism_ml_cv_manifest.json'
OUTPUTS={name:T/f'worldmove_mechanism_{name}.csv' for name in [
    'between_primary','between_sensitivity','between_loco','between_fitted',
    'within_primary','within_sensitivity','within_loco','within_loco_predictions','within_residuals']}


def require(condition,message):
    if not condition:raise RuntimeError(message)


def format_rows(fit,names,model_id,method,df_ref,n,sdx,sdy,pearson=None,spearman=None):
    vals=np.asarray(fit['beta']);cov=np.asarray(fit['cov']);se=np.sqrt(np.diag(cov));q=student_t.ppf(.975,df_ref)
    out=[]
    for j,name in enumerate(names):
        sx=sdx.get(name,np.nan)
        out.append(dict(model=model_id,term=name,estimate=vals[j],se=se[j],ci_low=vals[j]-q*se[j],
                        ci_high=vals[j]+q*se[j],ci_method=method,ci_t_df=df_ref,
                        effect_per_predictor_sd=vals[j]*sx if np.isfinite(sx) else np.nan,
                        fully_standardized_beta=vals[j]*sx/sdy if np.isfinite(sx) and sdy>0 else np.nan,
                        pearson_with_outcome=(pearson or {}).get(name,np.nan),
                        spearman_with_outcome=(spearman or {}).get(name,np.nan),
                        n=n,r2=fit['r2'],adjusted_r2=fit.get('adjusted_r2',np.nan),
                        within_r2=fit.get('within_r2',np.nan),rss=fit['rss'],
                        rmse=np.sqrt(fit['rss']/n),outcome_mean=fit['mean_y'],
                        outcome_sd=sdy,non_iid_approximation_note='Conditional model-based interval; 33-city nonprobability frame and product error not included'))
    return out


def fit_between(df,features,ycol,model_id,ci=True):
    X=sm.add_constant(df[features].to_numpy(dtype=float),has_constant='add')
    y=df[ycol].to_numpy(dtype=float)
    require(np.linalg.matrix_rank(X)==X.shape[1],model_id+' rank failure')
    result=sm.OLS(y,X).fit()
    robust=result.get_robustcov_results(cov_type='HC3',use_t=True) if ci else result
    pred=np.asarray(result.predict(X));resid=y-pred
    return dict(beta=np.asarray(result.params),cov=np.asarray(robust.cov_params()),
                r2=float(result.rsquared),adjusted_r2=float(result.rsquared_adj),
                rss=float(np.dot(resid,resid)),mean_y=float(y.mean()),pred=pred,
                residual=resid,result=result)


def between_rows(df,features,ycol,model_id,pearson_descriptive=False):
    fit=fit_between(df,features,ycol,model_id)
    sdx={name:float(df[name].std(ddof=1)) for name in features}
    corr_p={};corr_s={}
    if pearson_descriptive:
        for name in features:
            corr_p[name]=float(pearsonr(df[name],df[ycol]).statistic)
            corr_s[name]=float(spearmanr(df[name],df[ycol]).statistic)
    rows=format_rows(fit,['intercept']+features,model_id,'HC3',len(df)-len(features)-1,len(df),
                     sdx,float(df[ycol].std(ddof=1)),corr_p,corr_s)
    return rows,fit


def fit_within(df,features,ycol,model_id):
    # All supplied time-varying predictors are city-centered. Month indicators
    # are also city-centered before fitting; no held-city intercept is estimated.
    d=df.copy()
    for name in features:
        d[name]=d[name]-d.groupby('city',sort=False)[name].transform('mean')
    y=(d[ycol]-d.groupby('city',sort=False)[ycol].transform('mean')).to_numpy(dtype=float)
    X=d[features].to_numpy(dtype=float)
    require(np.linalg.matrix_rank(X)==X.shape[1],model_id+' rank failure')
    result=sm.OLS(y,X).fit()
    pred=np.asarray(result.predict(X));resid=y-pred
    time=pd.Categorical(d.date,categories=sorted(d.date.unique()),ordered=True).codes
    require(len(set(time))==153,model_id+' date count')
    cov=np.asarray(cov_nw_groupsum(result,nlags=7,time=time,use_correction='hac'))
    require(np.isfinite(cov).all() and (np.diag(cov)>=0).all(),model_id+' DK covariance invalid')
    tss=float(np.dot(y,y));rss=float(np.dot(resid,resid));r2=1-rss/tss
    return dict(beta=np.asarray(result.params),cov=cov,r2=r2,within_r2=r2,rss=rss,mean_y=float(d[ycol].mean()),
                pred=pred,residual=resid,y_anomaly=y,result=result,centered_X=X)


def within_rows(df,features,ycol,model_id):
    fit=fit_within(df,features,ycol,model_id)
    y_anom=df[ycol]-df.groupby('city',sort=False)[ycol].transform('mean')
    sdx={name:float((df[name]-df.groupby('city',sort=False)[name].transform('mean')).std(ddof=1)) for name in features}
    rows=format_rows(fit,features,model_id,'Driscoll-Kraay Bartlett 7d hac',32,len(df),sdx,float(y_anom.std(ddof=1)))
    return rows,fit


def read_outcomes(cfg):
    daily=pd.read_csv(T/'worldmove_formal_city_date.csv',usecols=['city','date','analysis_variant','mean_abs_delta_ug_m3'])
    pooled=pd.read_csv(T/'worldmove_formal_city_may_sep.csv',usecols=['city','date_equal_mean_abs_delta_ug_m3'])
    high=pd.read_csv(T/'worldmove_formal_high_confidence.csv',usecols=['city','date','aggregation','mean_abs_delta_ug_m3','date_equal_mean_abs_delta_ug_m3'])
    assert len(daily)==5049 and len(pooled)==33
    assert set(daily.analysis_variant)=={'primary'}
    high_daily=high[high.aggregation=='daily'][['city','date','mean_abs_delta_ug_m3']].copy()
    high_pooled=high[high.aggregation=='may_september_pooled'][['city','date_equal_mean_abs_delta_ug_m3']].copy()
    assert len(high_daily)==5049 and len(high_pooled)==33
    for df in (daily,high_daily):
        assert not df.duplicated(['city','date']).any()
        assert set(df.date)==set(cfg['formal_frame']['dates'])
        assert df.groupby('city').size().eq(153).all()
        assert np.isfinite(df.mean_abs_delta_ug_m3).all()
    for df in (pooled,high_pooled):
        assert not df.duplicated('city').any()
        assert set(df.city)==set(cfg['formal_frame']['cities'])
    # No outcome re-estimation; this only checks the already-defined aggregation.
    for low,agg,label in [(daily,pooled,'primary'),(high_daily,high_pooled,'high')]:
        check=low.groupby('city',sort=False).mean_abs_delta_ug_m3.mean().reset_index().merge(agg,on='city',validate='one_to_one')
        require(np.max(np.abs(check.mean_abs_delta_ug_m3-check.date_equal_mean_abs_delta_ug_m3))<1e-10,
                label+' pooled outcome/daily aggregation mismatch')
    return daily,pooled,high_daily,high_pooled


def main():
    if MODEL_MANIFEST.exists() or CV_MANIFEST.exists() or any(p.exists() for p in OUTPUTS.values()):
        raise FileExistsError('Model/LOCO artifacts already exist; refusing overwrite')
    predictor_report=predictor_audit()  # hard gate before any formal Y values are parsed
    cfg=json.loads((P/'worldmove_mechanism_design_config.json').read_text())
    cities=cfg['formal_frame']['cities'];dates=cfg['formal_frame']['dates']
    # Freeze all 33 grouped folds and their five names before any model fit.
    fold_manifest={'status':'ML_FOLDS_FROZEN_BEFORE_FITTING','created_at_utc':datetime.now(timezone.utc).isoformat(),
                   'predictor_manifest_sha256':sha(P/'worldmove_mechanism_predictor_manifest.json'),
                   'exact_config_sha256':sha(P/'worldmove_mechanism_design_config.json'),
                   'feature_names':cfg['xgboost']['predictors'],
                   'folds':[{'held_out_city':c,'train_cities':[x for x in cities if x!=c],
                             'test_dates':dates} for c in cities],
                   'group':'city','no_random_row_split':True}
    with CV_MANIFEST.open('x',encoding='utf-8') as f:json.dump(fold_manifest,f,ensure_ascii=False,indent=2)
    daily_y,pooled_y,high_daily_y,high_pooled_y=read_outcomes(cfg)
    city=pd.read_csv(T/'worldmove_mechanism_predictors_city.csv')
    day=pd.read_csv(T/'worldmove_mechanism_predictors_city_date.csv')
    primary_city=city[city.variant=='primary'].merge(pooled_y,on='city',validate='one_to_one')
    high_city=city[city.variant=='high_confidence'].merge(high_pooled_y,on='city',validate='one_to_one')
    primary_day=day[day.variant=='primary'].merge(daily_y[['city','date','mean_abs_delta_ug_m3']],on=['city','date'],validate='one_to_one')
    high_day=day[day.variant=='high_confidence'].merge(high_daily_y,on=['city','date'],validate='one_to_one')
    assert list(primary_city.city)==cities and list(high_city.city)==cities
    assert len(primary_day)==len(high_day)==5049
    yb='date_equal_mean_abs_delta_ug_m3';yw='mean_abs_delta_ug_m3'
    core_b=['T','E','Hbar'];core_w=['Hw','Aw']
    between_primary=[];between_sensitivity=[];between_loco=[];between_fitted=[]
    primary_rows,bfit=between_rows(primary_city,core_b,yb,'B_full',True)
    between_primary+=primary_rows
    h_rows,hfit=between_rows(primary_city,['Hbar'],yb,'B_Hbar_only')
    between_primary+=h_rows
    null_rows,nfit=between_rows(primary_city,[],yb,'B_intercept_only')
    between_primary+=null_rows
    for df,features,model in [
        (high_city,core_b,'B1_high_confidence'),
        (primary_city,['T','E','H90bar'],'B2_P90_P10'),
        (primary_city,core_b+['log_released_grid_count'],'B4_geometry'),
        (primary_city[~primary_city.city.isin(['Shanghai','Tangshan','Guangzhou'])],core_b,'B5_fixed_mask_removed'),
        (primary_city,core_b+['Ubar'],'B6_background')]:
        rr,_=between_rows(df,features,yb,model)
        between_sensitivity+=rr
    for j,city_name in enumerate(cities):
        train=primary_city[primary_city.city!=city_name];test=primary_city[primary_city.city==city_name]
        fit=fit_between(train,core_b,yb,'B3_LOCO_'+city_name)
        prediction=float(np.r_[1.,test[core_b].to_numpy()[0]]@fit['beta'])
        observed=float(test[yb].iloc[0])
        for k,term in enumerate(['intercept']+core_b):
            between_loco.append(dict(held_out_city=city_name,term=term,estimate=fit['beta'][k],
                                     full_model_estimate=bfit['beta'][k],
                                     coefficient_change=fit['beta'][k]-bfit['beta'][k],
                                     held_out_prediction=prediction,held_out_observed=observed,
                                     held_out_error=prediction-observed,held_out_abs_error=abs(prediction-observed),
                                     training_cities=32))
    infl=bfit['result'].get_influence()
    for j,c in enumerate(cities):
        between_fitted.append(dict(city=c,observed=primary_city[yb].iloc[j],fitted=bfit['pred'][j],
                                   residual=bfit['residual'][j],leverage=infl.hat_matrix_diag[j],
                                   cooks_distance=infl.cooks_distance[0][j]))

    within_primary=[];within_sensitivity=[];within_loco=[];within_loco_pred=[];within_resid=[]
    wrows,wfit=within_rows(primary_day,core_w,yw,'W_full')
    within_primary+=wrows
    reduced_rows,wreduced=within_rows(primary_day,['Hw'],yw,'W_H_only')
    within_primary+=reduced_rows
    for df,features,model in [
        (high_day,core_w,'W1_high_confidence'),
        (primary_day,['H90w','A90w'],'W2_P90_P10'),
        (primary_day[~primary_day.city.isin(['Shanghai','Tangshan','Guangzhou'])],core_w,'W4_fixed_mask_removed')]:
        rr,_=within_rows(df,features,yw,model)
        within_sensitivity+=rr
    background=primary_day.copy()
    for m,label in [('2019-06','June'),('2019-07','July'),('2019-08','August'),('2019-09','September')]:
        background[label]=(background.month==m).astype(float)
    bg_features=['Hw','Aw','Uw','June','July','August','September']
    bg_rows,_=within_rows(background,bg_features,yw,'W5_background_calendar_joint')
    within_sensitivity+=bg_rows
    for held in cities:
        train=primary_day[primary_day.city!=held]
        test=primary_day[primary_day.city==held].copy()
        # A common slope is learned from training-city anomalies. Test predictors
        # are already predictor-only centered; no held-out Y enters predictions.
        train_y=(train[yw]-train.groupby('city',sort=False)[yw].transform('mean')).to_numpy()
        train_x=train[core_w].to_numpy()
        beta=np.linalg.lstsq(train_x,train_y,rcond=None)[0]
        pred=test[core_w].to_numpy()@beta
        beta_reduced=np.linalg.lstsq(train_x[:,:1],train_y,rcond=None)[0]
        pred_reduced=test[['Hw']].to_numpy()@beta_reduced
        # Test Y enters only after predictions exist, exclusively for scoring.
        observed=test[yw].to_numpy();actual_anomaly=observed-observed.mean()
        error=pred-actual_anomaly
        reduced_error=pred_reduced-actual_anomaly
        for k,term in enumerate(core_w):
            within_loco.append(dict(held_out_city=held,term=term,estimate=beta[k],
                                    full_model_estimate=wfit['beta'][k],coefficient_change=beta[k]-wfit['beta'][k],
                                    held_city_anomaly_rmse=np.sqrt(np.mean(error**2)),
                                    held_city_H_only_rmse=np.sqrt(np.mean(reduced_error**2))))
        for date,prediction,reduced_prediction,actual in zip(test.date,pred,pred_reduced,actual_anomaly):
            within_loco_pred.append(dict(city=held,date=date,predicted_anomaly=prediction,
                                         H_only_predicted_anomaly=reduced_prediction,
                                         observed_anomaly_for_scoring_only=actual,error=prediction-actual,
                                         H_only_error=reduced_prediction-actual))
    for c in cities:
        r=wfit['residual'][primary_day.city.to_numpy()==c]
        within_resid.append(dict(city=c,residual_sd=float(np.std(r,ddof=1)),
                                 residual_mean=float(np.mean(r)),
                                 residual_lag1_corr=float(np.corrcoef(r[:-1],r[1:])[0,1]),
                                 residual_rmse=float(np.sqrt(np.mean(r*r)))))
    for name,rows in [('between_primary',between_primary),('between_sensitivity',between_sensitivity),
                      ('between_loco',between_loco),('between_fitted',between_fitted),
                      ('within_primary',within_primary),('within_sensitivity',within_sensitivity),
                      ('within_loco',within_loco),('within_loco_predictions',within_loco_pred),
                      ('within_residuals',within_resid)]:
        with OUTPUTS[name].open('x',newline='',encoding='utf-8') as f:
            pd.DataFrame(rows).to_csv(f,index=False,float_format='%.17g')
    model_manifest=dict(status='FORMAL_LOW_DIMENSIONAL_MODELS_COMPLETE',completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        script_sha256=sha(Path(__file__)),config_sha256=sha(P/'worldmove_mechanism_design_config.json'),
                        predictor_manifest_sha256=sha(P/'worldmove_mechanism_predictor_manifest.json'),
                        cv_manifest_sha256=sha(CV_MANIFEST),
                        outcome_file_sha256={str((T/name).relative_to(ROOT)):sha(T/name) for name in [
                            'worldmove_formal_city_date.csv','worldmove_formal_city_may_sep.csv','worldmove_formal_high_confidence.csv']},
                        output_sha256={str(path.relative_to(ROOT)):sha(path) for path in OUTPUTS.values()},
                        predictor_independent_audit=predictor_report,
                        software_versions={'numpy':np.__version__,'pandas':pd.__version__,'statsmodels':statsmodels.__version__},
                        fit_counts={'between_primary':3,'between_sensitivity':5,'between_loco':33,
                                    'within_primary':2,'within_sensitivity':4,'within_loco':33},
                        ml_status='PENDING')
    with MODEL_MANIFEST.open('x',encoding='utf-8') as f:json.dump(model_manifest,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('FORMAL_MODELS_COMPLETE',json.dumps({'between_adj_r2':bfit['adjusted_r2'],
                                                'within_r2':wfit['within_r2'],
                                                'within_alignment_increment_r2':wfit['within_r2']-wreduced['within_r2']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
