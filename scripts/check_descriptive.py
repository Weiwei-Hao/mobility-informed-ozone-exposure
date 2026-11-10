#!/usr/bin/env python3
"""Reconstruct the paper's equal-city descriptive summaries from supplied data."""
from pathlib import Path
import argparse,json,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from release_support import DEFAULT_DATA

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,default=DEFAULT_DATA);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    data=a.data_root/'tables'
    daily=pd.read_csv(data/'worldmove_formal_city_date.csv')
    period=pd.read_csv(data/'worldmove_formal_city_may_sep.csv')
    months=pd.read_csv(data/'worldmove_formal_city_month.csv')
    assert len(daily)==5049 and daily.city.nunique()==33 and daily.date.nunique()==153
    assert len(period)==33 and len(months)==165
    mean=daily.groupby('city',sort=False)[['mean_abs_delta_ug_m3','mean_delta_ug_m3']].mean()
    target=period.set_index('city').loc[mean.index]
    err=float(np.max(np.abs(mean.mean_abs_delta_ug_m3-target.date_equal_mean_abs_delta_ug_m3)))
    assert err<1e-10
    values={'equal_city_signed_ug_m3':float(mean.mean_delta_ug_m3.mean()),'equal_city_absolute_ug_m3':float(mean.mean_abs_delta_ug_m3.mean()),
        'absolute_city_min_ug_m3':float(mean.mean_abs_delta_ug_m3.min()),'absolute_city_max_ug_m3':float(mean.mean_abs_delta_ug_m3.max()),
        'cities_daily_absolute_SD_exceeds_between_city_SD':int((daily.groupby('city').mean_abs_delta_ug_m3.std(ddof=1)>mean.mean_abs_delta_ug_m3.std(ddof=1)).sum())}
    checks=[f'{values["equal_city_signed_ug_m3"]:.3f}'=='0.062',f'{values["equal_city_absolute_ug_m3"]:.3f}'=='1.389',
            values['cities_daily_absolute_SD_exceeds_between_city_SD']==29]
    report={'passed':all(checks),'cities':33,'dates':153,'city_date_rows':5049,'monthly_rows':165,'period_rows':33,
            'daily_to_period_max_abs_difference':err,'values':values,'model_fitting':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
