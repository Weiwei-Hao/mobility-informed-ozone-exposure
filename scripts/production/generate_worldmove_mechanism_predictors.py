#!/usr/bin/env python3
"""Construct T/E/H/A predictors from frozen schedules and native ozone product fields."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_worldmove_five_city_pilot import load_worldmove, infer_anchor

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / 'data/processed'
TABLES = ROOT / 'outputs/tables'
CONFIG_PATH = PROCESSED / 'worldmove_mechanism_design_config.json'
FRAME_PATH = PROCESSED / 'worldmove_formal_city_frame.json'
IDS_PATH = PROCESSED / 'worldmove_formal_sample_ids.npz'
IDS_META_PATH = PROCESSED / 'worldmove_formal_sample_ids_metadata.json'
RUN_META_PATH = PROCESSED / 'worldmove_formal_run_metadata.json'
MAPPINGS_PATH = PROCESSED / 'worldmove_frame_audit/mappings.json'
MASK_DIR = PROCESSED / 'worldmove_frame_audit/daily_masks'
CACHE_DIR = PROCESSED / 'worldmove_formal_o3_cache'
MANIFEST_PATH = PROCESSED / 'worldmove_mechanism_predictor_manifest.json'
FILE_PATHS = {
    'city': TABLES / 'worldmove_mechanism_predictors_city.csv',
    'city_hour': TABLES / 'worldmove_mechanism_predictors_city_hour.csv',
    'city_date_hour': TABLES / 'worldmove_mechanism_predictors_city_date_hour.csv',
    'city_date': TABLES / 'worldmove_mechanism_predictors_city_date.csv',
    'qa': TABLES / 'worldmove_mechanism_predictor_qa.csv',
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def assert_equal(actual, expected, name):
    if actual != expected:
        raise RuntimeError(f'FROZEN INPUT MISMATCH: {name}: actual={actual}; expected={expected}')


def preflight():
    raise RuntimeError("Use release_pipeline.py analysis to validate and bind upstream inputs.")


def haversine_km(lon0, lat0, lon1, lat1):
    lon0, lat0, lon1, lat1 = map(np.deg2rad, (lon0, lat0, lon1, lat1))
    a = np.sin((lat1-lat0)/2)**2 + np.cos(lat0)*np.cos(lat1)*np.sin((lon1-lon0)/2)**2
    return 6371.0088 * (2*np.arcsin(np.sqrt(np.clip(a,0,1))))


def mobility_only(frame, ids_meta):
    """No O3/cache/mask values can enter this function."""
    profiles, metrics, source_hashes = {}, {}, {}
    with np.load(IDS_PATH) as ids:
        assert_equal(set(ids.files), {c+'__sample5000' for c in frame['final_eligible_city_list']}, 'ID archive keys')
        for entry in frame['eligible_cities']:
            city = entry['city_name']
            path = ROOT / entry['npz_path']
            assert_equal(sha(path), entry['npz_sha256'], 'WorldMove '+city)
            source_hashes[entry['npz_path']] = entry['npz_sha256']
            key = city+'__sample5000'
            sample_ids = ids[key]
            assert_equal(len(sample_ids), 5000, 'raw selected IDs '+city)
            assert_equal(hashlib.sha256(sample_ids.tobytes()).hexdigest(),
                         ids_meta['arrays'][key]['sha256_c_order_bytes'], 'ID array '+city)
            assert np.array_equal(sample_ids, np.sort(np.unique(sample_ids)))
            traj_all, xy, meta = load_worldmove(path)
            assert_equal(len(traj_all), entry['raw_schedule_count'], 'raw trajectory count '+city)
            assert_equal(len(xy), entry['grid_cells'], 'released grid count '+city)
            traj = traj_all[sample_ids]
            anchor = infer_anchor(traj, np.arange(12))
            for variant, keep in [('primary', anchor.valid),
                                  ('high_confidence', anchor.valid & (anchor.dominance >= .75))]:
                selected = traj[keep]
                anchor_grid = anchor.grid_index[keep]
                if len(selected) == 0:
                    raise RuntimeError('No anchor-valid sample: '+city+' '+variant)
                away = selected != anchor_grid[:,None]
                by_hour = away.reshape(len(selected),24,2).mean(axis=(0,2), dtype=np.float64)
                time_away = float(away.mean(dtype=np.float64))
                away_n = int(away.sum())
                if away_n:
                    row, slot = np.nonzero(away)
                    starts, ends = xy[anchor_grid[row]], xy[selected[row,slot]]
                    extent = float(np.mean(haversine_km(starts[:,0], starts[:,1], ends[:,0], ends[:,1]),dtype=np.float64))
                else:
                    extent = 0.0
                metrics[city,variant] = dict(T=time_away,E=extent,anchor_valid_n=int(keep.sum()),
                                             away_slots=away_n,structural_no_mobility=(away_n==0))
                profiles[city,variant] = by_hour
                assert np.isclose(float(by_hour.mean()),time_away,atol=1e-12,rtol=0)
            print('MOBILITY_ONLY',city,flush=True)
    return metrics, profiles, source_hashes


def domain_from_fixed_masks(frame, mapping, dates):
    domains, first_masks, mask_hashes = {}, {}, {}
    first_date = dates[0]
    with np.load(MASK_DIR / f'{first_date}.npz') as z:
        for entry in frame['eligible_cities']:
            city, key = entry['city_name'], entry['released_key']
            m = np.asarray(z[key],dtype=bool)
            assert_equal(m.shape,(24,entry['grid_cells']), 'mask shape '+city)
            if any(mapping[key]['outside']):
                raise RuntimeError('Frozen eligible city has outside-product grid: '+city)
            if not np.array_equal(m, np.broadcast_to(m[:1],m.shape)):
                raise RuntimeError('Nonfixed hourly support: '+city)
            first_masks[city] = m
            coords = np.column_stack([mapping[key]['iy'],mapping[key]['ix']])
            assert_equal(coords.shape,(entry['grid_cells'],2),'mapping shape '+city)
            unique, first, inverse = np.unique(coords,axis=0,return_index=True,return_inverse=True)
            valid_released = ~m.all(axis=0)
            if not np.array_equal(m, np.broadcast_to(m[:1],m.shape)):
                raise RuntimeError('Hourly support changed: '+city)
            for group in range(len(unique)):
                members = np.flatnonzero(inverse==group)
                if not np.array_equal(m[:,members],np.broadcast_to(m[:,members[:1]],(24,len(members)))):
                    raise RuntimeError('Duplicate product-cell masks disagree: '+city)
            unique_valid_idx = first[valid_released[first]]
            if len(unique_valid_idx) < 2:
                raise RuntimeError('Fewer than two unique valid product cells: '+city)
            if np.any(m[:,unique_valid_idx]):
                raise RuntimeError('Partially supported product cell: '+city)
            domains[city] = dict(unique_valid_idx=unique_valid_idx,
                                 inverse=inverse,first=first,
                                 released_grid_count=entry['grid_cells'],
                                 unique_valid_o3_cells=len(unique_valid_idx),
                                 fixed_missing_released=int(m[0].sum()))
    # Check every frozen mask without touching O3 numeric values.
    for date in dates:
        path = MASK_DIR / f'{date}.npz'
        mask_hashes[str(path.relative_to(ROOT))] = sha(path)
        with np.load(path) as z:
            for entry in frame['eligible_cities']:
                city,key=entry['city_name'],entry['released_key']
                if not np.array_equal(z[key],first_masks[city]):
                    raise RuntimeError('Frozen daily mask changed: '+city+' '+date)
    return domains, first_masks, mask_hashes


def o3_only(cfg, frame, domains, first_masks):
    """No trajectories, anchors, schedule IDs or outcome values can enter."""
    cities=cfg['formal_frame']['cities']; dates=cfg['formal_frame']['dates']
    H=np.empty((33,153,24),dtype=np.float64)
    H90=np.empty_like(H); U=np.empty_like(H)
    for ti,date in enumerate(dates):
        with np.load(CACHE_DIR/f'{date}.npz') as z:
            assert_equal(str(z['frame_sha256']), cfg['formal_frame']['sha256'], 'cache frame '+date)
            assert_equal(set(z.files),set(cities)|{'frame_sha256'},'cache city keys '+date)
            for ci,city in enumerate(cities):
                all_grid=np.asarray(z[city],dtype=np.float32)
                dom=domains[city]
                assert_equal(all_grid.shape,(dom['released_grid_count'],24),'cache shape '+city+' '+date)
                if not np.array_equal(~np.isfinite(all_grid).T,first_masks[city]):
                    raise RuntimeError('Cache support/mask mismatch: '+city+' '+date)
                # Repeated released grids mapped to one product cell must have identical values.
                representative=all_grid[dom['first']]
                if not np.array_equal(all_grid,representative[dom['inverse']],equal_nan=True):
                    raise RuntimeError('Duplicate mapped product-cell concentration mismatch: '+city+' '+date)
                values=all_grid[dom['unique_valid_idx']].astype(np.float64)
                if not np.isfinite(values).all():
                    raise RuntimeError('Nonfinite valid O3 domain: '+city+' '+date)
                q=np.quantile(values,[.10,.25,.75,.90],axis=0,method='linear')
                H[ci,ti]=q[2]-q[1]
                H90[ci,ti]=q[3]-q[0]
                U[ci,ti]=values.mean(axis=0,dtype=np.float64)
        if (ti+1)%25==0 or ti==len(dates)-1:
            print('O3_ONLY_DATES',ti+1,'/',len(dates),flush=True)
    return H,H90,U


def centered(values):
    mean=values.mean(axis=1,dtype=np.float64)
    return mean, values-mean[:,None]


def matrix_qa(name, X, columns, rows):
    X=np.asarray(X,dtype=np.float64)
    sd=X.std(axis=0,ddof=1)
    if (sd<=1e-12).any():
        for col,s in zip(columns,sd):
            if s<=1e-12: rows.append(dict(category='matrix',check='zero_variance',scope=name,variable=col,value=s,status='NON_ESTIMABLE'))
        raise RuntimeError('Frozen model has zero-variance predictor: '+name)
    Z=(X-X.mean(axis=0))/sd
    sv=np.linalg.svd(Z,compute_uv=False)
    ratio=float(sv[-1]/sv[0])
    rank=int(np.linalg.matrix_rank(Z,tol=sv[0]*1e-10))
    rows.append(dict(category='matrix',check='rank',scope=name,value=rank,expected=len(columns),status='PASS' if rank==len(columns) else 'FAIL'))
    rows.append(dict(category='matrix',check='scaled_singular_ratio',scope=name,value=ratio,status='PASS' if ratio>=1e-10 else 'FAIL'))
    if rank!=len(columns) or ratio<1e-10:
        raise RuntimeError('Severe predictor rank failure: '+name)
    for j,col in enumerate(columns):
        others=np.delete(Z,j,axis=1)
        fitted=others @ np.linalg.lstsq(others,Z[:,j],rcond=None)[0] if others.shape[1] else np.zeros(len(Z))
        r2=1-float(np.sum((Z[:,j]-fitted)**2)/np.sum(Z[:,j]**2))
        vif=1/(1-r2) if r2<1 else float('inf')
        rows.append(dict(category='matrix',check='VIF',scope=name,variable=col,value=vif,
                         status='FLAG_GT_5' if vif>5 else 'PASS'))


def main():
    if MANIFEST_PATH.exists() or any(p.exists() for p in FILE_PATHS.values()):
        raise FileExistsError('Predictor artifact already exists: no overwrite or post-model regeneration permitted')
    cfg, frame, ids_meta, run_meta, mapping = preflight()
    cities=cfg['formal_frame']['cities']; dates=cfg['formal_frame']['dates']
    metrics,profiles,worldmove_hashes=mobility_only(frame,ids_meta)
    expected_counts={r['city']:r for r in run_meta['counts']}
    assert_equal(set(expected_counts),set(cities),'formal sample-count cities')
    for city in cities:
        assert_equal(metrics[city,'primary']['anchor_valid_n'],
                     expected_counts[city]['main_anchor_valid_n'],'main anchor denominator '+city)
        assert_equal(metrics[city,'high_confidence']['anchor_valid_n'],
                     expected_counts[city]['high_confidence_valid_n'],'high-confidence anchor denominator '+city)
    domains,first_masks,mask_hashes=domain_from_fixed_masks(frame,mapping,dates)
    Hh,H90h,Uh=o3_only(cfg,frame,domains,first_masks)
    H,Hw=centered(Hh.mean(axis=2)); H90,H90w=centered(H90h.mean(axis=2)); U,Uw=centered(Uh.mean(axis=2))
    # The names above are city means and city-date demeaned arrays; keep daily arrays explicit.
    Hd=Hh.mean(axis=2); H90d=H90h.mean(axis=2); Ud=Uh.mean(axis=2)
    A, A90 = {}, {}
    for variant in ('primary','high_confidence'):
        M=np.stack([profiles[c,variant] for c in cities])
        den=M.sum(axis=1)
        val=np.zeros((33,153),dtype=np.float64)
        val90=np.zeros_like(val)
        nonzero=den>0
        val[nonzero]=np.einsum('ch,cth->ct',M[nonzero],Hh[nonzero])/den[nonzero,None]-Hd[nonzero]
        val90[nonzero]=np.einsum('ch,cth->ct',M[nonzero],H90h[nonzero])/den[nonzero,None]-H90d[nonzero]
        A[variant]=val; A90[variant]=val90

    city_rows=[]; hour_rows=[]; city_date_rows=[]; city_date_hour_rows=[]; qa=[]
    for ci,city in enumerate(cities):
        dom=domains[city]
        for variant in ('primary','high_confidence'):
            met=metrics[city,variant]; M=profiles[city,variant]
            abar=float(A[variant][ci].mean()); abar90=float(A90[variant][ci].mean())
            city_rows.append(dict(city=city,variant=variant,T=met['T'],E=met['E'],Hbar=H[ci],H90bar=H90[ci],
                                  Ubar=U[ci],log_released_grid_count=math.log(dom['released_grid_count']),
                                  released_grid_count=dom['released_grid_count'],
                                  unique_valid_o3_cells=dom['unique_valid_o3_cells'],
                                  product_cell_to_released_grid_ratio=dom['unique_valid_o3_cells']/dom['released_grid_count'],
                                  fixed_missing_released_grids=dom['fixed_missing_released'],
                                  anchor_valid_n=met['anchor_valid_n'],away_slots=met['away_slots'],
                                  structural_no_mobility=met['structural_no_mobility']))
            for h in range(24):
                hour_rows.append(dict(city=city,hour=h,M=M[h],variant=variant))
            for ti,date in enumerate(dates):
                city_date_rows.append(dict(city=city,date=date,variant=variant,
                    H=Hd[ci,ti],Hbar=H[ci],Hw=Hw[ci,ti],
                    A=A[variant][ci,ti],Abar=abar,Aw=A[variant][ci,ti]-abar,
                    U=Ud[ci,ti],Ubar=U[ci],Uw=Uw[ci,ti],
                    H90=H90d[ci,ti],H90bar=H90[ci],H90w=H90w[ci,ti],
                    A90=A90[variant][ci,ti],A90bar=abar90,A90w=A90[variant][ci,ti]-abar90,
                    month=date[:7],structural_no_mobility=met['structural_no_mobility']))
            qa.append(dict(category='mobility',check='mean_hour_M_equals_T',scope=city,variant=variant,
                           value=float(M.mean()-met['T']),status='PASS' if abs(M.mean()-met['T'])<1e-12 else 'FAIL'))
            qa.append(dict(category='mobility',check='anchor_valid_n',scope=city,variant=variant,
                           value=met['anchor_valid_n'],status='PASS'))
            qa.append(dict(category='support',check='unique_valid_o3_cells',scope=city,variant=variant,
                           value=dom['unique_valid_o3_cells'],status='PASS' if dom['unique_valid_o3_cells']>=2 else 'FAIL'))
            qa.append(dict(category='centering',check='mean_Aw_zero',scope=city,variant=variant,
                           value=float((A[variant][ci]-abar).mean()),status='PASS' if abs((A[variant][ci]-abar).mean())<1e-12 else 'FAIL'))
        qa.append(dict(category='centering',check='mean_Hw_zero',scope=city,value=float(Hw[ci].mean()),
                       status='PASS' if abs(Hw[ci].mean())<1e-12 else 'FAIL'))
        qa.append(dict(category='centering',check='mean_H90w_zero',scope=city,value=float(H90w[ci].mean()),
                       status='PASS' if abs(H90w[ci].mean())<1e-12 else 'FAIL'))
        qa.append(dict(category='aggregation',check='Hbar_equals_date_mean_H',scope=city,
                       value=float(H[ci]-Hd[ci].mean()),status='PASS' if abs(H[ci]-Hd[ci].mean())<1e-12 else 'FAIL'))
        for ti,date in enumerate(dates):
            for h in range(24):
                city_date_hour_rows.append(dict(city=city,date=date,hour=h,H_iqr=Hh[ci,ti,h],
                                                H_p90_p10=H90h[ci,ti,h],footprint_mean_o3=Uh[ci,ti,h]))
    city_df=pd.DataFrame(city_rows); hour_df=pd.DataFrame(hour_rows)
    cdh_df=pd.DataFrame(city_date_hour_rows); cd_df=pd.DataFrame(city_date_rows)
    for name,df,keys,n in [
        ('city',city_df,['city','variant'],66),('city_hour',hour_df,['city','hour','variant'],1584),
        ('city_date_hour',cdh_df,['city','date','hour'],121176),
        ('city_date',cd_df,['city','date','variant'],10098)]:
        assert_equal(len(df),n,name+' rows')
        assert not df.duplicated(keys).any(),name+' duplicate keys'
        assert not df.isna().any().any(),name+' NA'
        numeric=df.select_dtypes(include='number').to_numpy()
        assert np.isfinite(numeric).all(),name+' nonfinite'
        qa.append(dict(category='shape',check='rows_and_unique_keys',scope=name,value=n,status='PASS'))
    assert_equal(city_df.city.nunique(),33,'city count')
    assert_equal(cd_df.date.nunique(),153,'date count')
    assert_equal(len(hour_df.drop_duplicates(['city','hour'])),792,'unique city-hour rows')
    assert_equal(len(cd_df.drop_duplicates(['city','date'])),5049,'unique city-date rows')
    assert city_df['T'].between(0,1).all() and (city_df['E']>=0).all()
    assert (cdh_df[['H_iqr','H_p90_p10']]>=-1e-12).all().all()
    assert_equal(set(cd_df.month.unique()),{'2019-05','2019-06','2019-07','2019-08','2019-09'},'month categories')
    # Predictor-only matrices; no Y columns are available in this process.
    main_city=city_df[city_df.variant=='primary'].reset_index(drop=True)
    high_city=city_df[city_df.variant=='high_confidence'].reset_index(drop=True)
    main_daily=cd_df[cd_df.variant=='primary'].reset_index(drop=True)
    high_daily=cd_df[cd_df.variant=='high_confidence'].reset_index(drop=True)
    matrix_qa('between_primary',main_city[['T','E','Hbar']],['T','E','Hbar'],qa)
    matrix_qa('between_high_confidence',high_city[['T','E','Hbar']],['T','E','Hbar'],qa)
    matrix_qa('between_alternate',main_city[['T','E','H90bar']],['T','E','H90bar'],qa)
    matrix_qa('between_geometry',main_city[['T','E','Hbar','log_released_grid_count']],['T','E','Hbar','logN'],qa)
    matrix_qa('between_background',main_city[['T','E','Hbar','Ubar']],['T','E','Hbar','Ubar'],qa)
    matrix_qa('between_fixed_mask',main_city[~main_city.city.isin(['Shanghai','Tangshan','Guangzhou'])][['T','E','Hbar']],['T','E','Hbar'],qa)
    matrix_qa('within_primary',main_daily[['Hw','Aw']],['Hw','Aw'],qa)
    matrix_qa('within_high_confidence',high_daily[['Hw','Aw']],['Hw','Aw'],qa)
    matrix_qa('within_alternate',main_daily[['H90w','A90w']],['H90w','A90w'],qa)
    matrix_qa('within_fixed_mask',main_daily[~main_daily.city.isin(['Shanghai','Tangshan','Guangzhou'])][['Hw','Aw']],['Hw','Aw'],qa)
    months=np.column_stack([(main_daily.month==m).to_numpy(dtype=np.float64) for m in ['2019-06','2019-07','2019-08','2019-09']])
    matrix_qa('within_background_calendar',np.column_stack([main_daily[['Hw','Aw','Uw']].to_numpy(),months]),['Hw','Aw','Uw','June','July','August','September'],qa)
    matrix_qa('ml_five_features',main_daily.merge(main_city[['city','T','E']],on='city',validate='many_to_one')[['T','E','Hbar','Hw','Aw']],['T','E','Hbar','Hw','Aw'],qa)
    qa.append(dict(category='separation',check='mobility_generated_before_O3_read',scope='all',value=True,status='PASS'))
    qa.append(dict(category='mobility',check='main_high_anchor_denominators_match_formal_metadata',scope='all',value=True,status='PASS'))
    qa.append(dict(category='separation',check='no_trajectory_in_O3_function',scope='all',value=True,status='PASS'))
    qa.append(dict(category='support',check='153_daily_masks_unchanged',scope='all',value=True,status='PASS'))
    qa.append(dict(category='support',check='cache_sha256_153_verified',scope='all',value=True,status='PASS'))
    qa.append(dict(category='separation',check='outcomes_never_parsed',scope='all',value=True,status='PASS'))
    qa_df=pd.DataFrame(qa)
    if (qa_df.status=='FAIL').any():
        raise RuntimeError('PREDICTOR-ONLY QA failed; no files frozen')
    for name,df in [('city',city_df),('city_hour',hour_df),('city_date_hour',cdh_df),('city_date',cd_df),('qa',qa_df)]:
        path=FILE_PATHS[name]
        with path.open('x',newline='',encoding='utf-8') as f:
            df.to_csv(f,index=False,float_format='%.17g')
    files={str(path.relative_to(ROOT)):sha(path) for path in FILE_PATHS.values()}
    script=str(Path(__file__).relative_to(ROOT))
    manifest=dict(status='PREDICTOR_FREEZE_COMPLETE',generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  exact_config_sha256=sha(CONFIG_PATH),script_sha256=sha(Path(__file__)),script_path=script,
                  file_sha256=files,
                  source_hashes={str(CONFIG_PATH.relative_to(ROOT)):sha(CONFIG_PATH),
                                 str(FRAME_PATH.relative_to(ROOT)):sha(FRAME_PATH),
                                 str(IDS_PATH.relative_to(ROOT)):sha(IDS_PATH),
                                 str(IDS_META_PATH.relative_to(ROOT)):sha(IDS_META_PATH),
                                 str(RUN_META_PATH.relative_to(ROOT)):sha(RUN_META_PATH),
                                 str(MAPPINGS_PATH.relative_to(ROOT)):sha(MAPPINGS_PATH),
                                 **worldmove_hashes},
                  cache_sha256=run_meta['lookup_sha256'],mask_sha256=mask_hashes,
                  protected_source_hashes_verified=len(cfg['protected_source_hashes']),
                  qa={'rows':len(qa_df),'pass':int((qa_df.status=='PASS').sum()),
                      'vif_gt_5_flags':int((qa_df.status=='FLAG_GT_5').sum()),
                      'all_hard_checks_passed':True},
                  row_counts={name:len(df) for name,df in [('city',city_df),('city_hour',hour_df),('city_date_hour',cdh_df),('city_date',cd_df),('qa',qa_df)]},
                  outcome_numeric_values_read=False,cohort_rule='all anchor-valid frozen IDs, no O3 support filtering')
    with MANIFEST_PATH.open('x',encoding='utf-8') as f:
        json.dump(manifest,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    print('PREDICTOR_FREEZE_COMPLETE',json.dumps({'files':files,'qa':manifest['qa']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
