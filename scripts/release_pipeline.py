#!/usr/bin/env python3
"""Portable I/O orchestration around the preserved WorldMove production functions.

No downloads. Every execution writes to a new work directory. A needs licensed
upstream inputs and frozen IDs. No figure-generation route is included.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
from release_support import resolve_input, DEFAULT_DATA
DERIVED_ROOT = Path(os.environ.get("WORLDMOVE_DERIVED_ROOT", DEFAULT_DATA)).resolve()

PACKAGE = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def requirements(mode):
    if mode != 'A':
        raise ValueError('Only upstream scientific inputs (mode A) are supported here.')
    return json.loads((PACKAGE / 'config/input_inventory.json').read_text())['A']


def check_inputs(mode, input_root):
    missing, changed = [], []
    for r in requirements(mode):
        p = resolve_input(r['path'], input_root, DERIVED_ROOT)
        if p is None:
            missing.append(r['path'])
        elif r.get('sha256') and sha(p) != r['sha256']:
            changed.append(r['path'])
    return {'mode': mode, 'missing': missing, 'changed': changed,
            'ready': not missing and not changed}


def stage(mode, input_root, work):
    report = check_inputs(mode, input_root)
    if not report['ready']:
        raise SystemExit(json.dumps(report, indent=2))
    if work.exists():
        raise SystemExit('Choose a new work directory; existing directories are not overwritten.')
    if any(work == p or p in work.parents for p in [input_root.resolve(), DERIVED_ROOT, PACKAGE]):
        raise SystemExit('The work directory must be outside the input and code trees.')
    work.mkdir(parents=True)
    shutil.copytree(PACKAGE / 'scripts/production', work / 'scripts')
    for r in requirements(mode):
        src, dst = resolve_input(r['path'], input_root, DERIVED_ROOT), work / r['path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Read-only use of legal inputs. All generated outputs live in work.
        if '/raw/' in r['path']:
            dst.symlink_to(src)
        else:
            shutil.copyfile(src, dst)
    p = work / 'data/processed'
    p.mkdir(parents=True, exist_ok=True)
    for name in ['worldmove_formal_city_frame.json', 'worldmove_mechanism_design_config.json']:
        dst = p / name
        if not dst.exists():
            shutil.copyfile(PACKAGE / 'config' / name, dst)
    (work / 'outputs/tables').mkdir(parents=True, exist_ok=True)
    dump(work / 'release_input_check.json', report)
    sys.path.insert(0, str(work / 'scripts'))
    return work


def run_analysis(input_root, work, include_ml=False):
    """A: recompute 33-city analyses from supplied upstream fields and frozen IDs.

    Input hashes, schema and support checks precede numerical calculations.
    Invoke explicitly; this is never a packaging side effect.
    """
    import numpy as np
    from netCDF4 import Dataset
    stage('A', input_root, work)
    core = importlib.import_module('run_worldmove_five_city_pilot')
    formal = importlib.import_module('run_worldmove_formal_multicity')
    gen = importlib.import_module('generate_worldmove_mechanism_predictors')
    frame = json.loads((work / 'data/processed/worldmove_formal_city_frame.json').read_text())
    cfg_path = work / 'data/processed/worldmove_mechanism_design_config.json'
    cfg = json.loads(cfg_path.read_text())
    # Numerical settings are frozen; execution provenance is checked separately.
    cfg['protected_source_hashes'] = {}
    dump(cfg_path, cfg)
    cities = frame['final_eligible_city_list']
    dates = frame['formal_153_day_date_support']['dates']
    inventory = requirements('A')
    files = {d: [] for d in dates}
    for r in inventory:
        if r.get('date') in files:
            files[r['date']].append(work / r['path'])
    for d in dates:
        files[d].sort(key=lambda p: p.name)
        assert len(files[d]) == 24
    lat, lon, ncmeta, ncqa = core.validate_netcdf_inputs(files)
    assert ncmeta['shape'] == [3599, 4799]
    assert ncmeta['storage_dtype'] == 'uint16' and ncmeta['fill_value'] == 65535
    assert ncmeta['scale_factor'] == float(np.float32(.1)) and ncmeta['add_offset'] == 0
    ids_path = work / 'data/processed/worldmove_formal_sample_ids.npz'
    ids_meta = json.loads((work / 'data/processed/worldmove_formal_sample_ids_metadata.json').read_text())
    assert sha(ids_path) == ids_meta['sha256'] == cfg['frozen_sample_ids']['sha256']
    loaded, mapping, counts = {}, {}, []
    with np.load(ids_path) as ids:
        for e in frame['eligible_cities']:
            c, key = e['city_name'], e['released_key']
            source = work / e['npz_path']
            assert sha(source) == e['npz_sha256']
            tr, xy, meta = core.load_worldmove(source)
            assert len(tr) == e['raw_schedule_count'] and len(xy) == e['grid_cells']
            sample_ids = ids[c + '__sample5000']
            assert len(sample_ids) == len(np.unique(sample_ids)) == 5000
            assert hashlib.sha256(sample_ids.tobytes()).hexdigest() == ids_meta['arrays'][c+'__sample5000']['sha256_c_order_bytes']
            full = core.infer_anchor(tr, np.arange(12))
            assert full.valid.sum() == e['main_anchor_valid_count']
            sample = tr[sample_ids]
            anchor = core.infer_anchor(sample, np.arange(12))
            high = core.restricted_anchor(anchor, anchor.valid & (anchor.dominance >= .75))
            loaded[c] = dict(traj=sample, anchor=anchor, high=high, full_traj=tr, full_anchor=full, entry=e)
            iy, ix = core.nearest_axis_indices(lat, xy[:,1]), core.nearest_axis_indices(lon, xy[:,0])
            outside = (xy[:,0]<lon.min()) | (xy[:,0]>lon.max()) | (xy[:,1]<lat.min()) | (xy[:,1]>lat.max())
            assert not outside.any()
            mapping[c] = dict(iy=iy.tolist(), ix=ix.tolist(), released_key=key)
            counts.append(dict(city=c, release_raw_n=len(tr), sampled_raw_n=5000,
                               main_anchor_valid_n=int(anchor.valid.sum()), high_confidence_valid_n=int(high.valid.sum())))
    cache = work / 'data/processed/worldmove_formal_o3_cache'
    masks = work / 'data/processed/worldmove_frame_audit/daily_masks'
    cache.mkdir(parents=True); masks.mkdir(parents=True)
    fixed = {}
    for day in dates:
        arrays = {c: np.empty((len(m['ix']),24),np.float32) for c,m in mapping.items()}
        # Same hourly crop, mask decode and float32 cast as production extract_day.
        for hour, path in enumerate(files[day]):
            with Dataset(path) as ds:
                v = ds['O3']
                for c,m in mapping.items():
                    y,x = np.array(m['iy']),np.array(m['ix'])
                    y0,y1,x0,x1 = y.min(),y.max(),x.min(),x.max()
                    crop = v[y0:y1+1,x0:x1+1]
                    crop = np.asarray(crop.filled(np.nan) if np.ma.isMaskedArray(crop) else crop,dtype=np.float32)
                    arrays[c][:,hour] = crop[y-y0,x-x0]
        mask_arrays = {mapping[c]['released_key']: ~np.isfinite(a).T for c,a in arrays.items()}
        if not fixed:
            fixed = {k:v.copy() for k,v in mask_arrays.items()}
        assert all(np.array_equal(v,fixed[k]) for k,v in mask_arrays.items())
        np.savez_compressed(masks / (day+'.npz'), **mask_arrays)
        arrays['frame_sha256'] = np.array(sha(formal.FRAME))
        np.savez_compressed(cache / (day+'.npz'), **arrays)
    for c, dd in loaded.items():
        mask = fixed[dd['entry']['released_key']]
        a,tr = dd['full_anchor'],dd['full_traj']; valid = a.valid
        complete = ~(mask[:,a.grid_index[valid]].any(axis=0) | mask[np.arange(48)//2,tr[valid]].any(axis=1))
        rate = float(complete.mean())
        assert rate >= frame['support_threshold'] == .975
        assert abs(rate-dd['entry']['daily_full_pool_complete_pair_retention']) < 1e-12
        del dd['full_traj'],dd['full_anchor']
    dump(work / 'data/processed/worldmove_frame_audit/mappings.json',
         {m['released_key']:dict(iy=m['iy'],ix=m['ix'],outside=[False]*len(m['ix'])) for m in mapping.values()})
    daily, high, diurnal, qc = [], [], [], list(ncqa)
    for c in formal.REF + [x for x in cities if x not in formal.REF]:
        def lookup(day):
            with np.load(cache/(day+'.npz')) as z:
                return z[c]
        result = formal.calculate(c, loaded[c], dates, lookup)
        for target, rows in zip((daily,high,diurnal,qc),result):
            target.extend(rows)
    order = {c:i for i,c in enumerate(cities)}
    daily.sort(key=lambda r:(order[r['city']],r['date']))
    high.sort(key=lambda r:(order[r['city']],r['date']))
    monthly, pooled = formal.aggregates(daily,cities,'primary')
    hm,hp = formal.aggregates(high,cities,'high_confidence')
    hc = [dict(r,aggregation='daily') for r in high] + [dict(r,aggregation='monthly') for r in hm] + [dict(r,aggregation='may_september_pooled') for r in hp]
    for name,rows in [('city_date',daily),('city_month',monthly),('city_may_sep',pooled),('high_confidence',hc),('diurnal',diurnal),('qc',qc),('sample_counts',counts)]:
        formal.table(work/'outputs/tables'/f'worldmove_formal_{name}.csv',rows)
    run_meta = dict(frame_sha256=sha(formal.FRAME), sample_freeze=ids_meta,
                    lookup_sha256={p.name:sha(p) for p in cache.glob('*.npz')}, counts=counts,
                    netcdf_metadata=ncmeta, execution='portable release rerun; historical regression gates not rerun')
    dump(work/'data/processed/worldmove_formal_run_metadata.json',run_meta)
    predictor_mapping = json.loads((work/'data/processed/worldmove_frame_audit/mappings.json').read_text())
    # Bind the validated input context to the unchanged predictor calculations.
    gen.preflight = lambda: (cfg,frame,ids_meta,run_meta,predictor_mapping)
    gen.main()
    importlib.import_module('run_worldmove_mechanism_models').main()
    if include_ml:
        importlib.import_module('run_worldmove_mechanism_ml').main()
    dump(work/'release_execution.json',dict(mode='A',include_ml=include_ml,
         completed=True, note='New calculation; compare table values with the frozen candidate inventory. No scientific claim follows from software execution alone.'))




def main_cli():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    c = sub.add_parser('check-inputs'); c.add_argument('--mode',choices=['A'],required=True);c.add_argument('--input-root',type=Path,required=True)
    for name in ['analysis']:
        c = sub.add_parser(name);c.add_argument('--input-root',type=Path,required=True);c.add_argument('--work-dir',type=Path,required=True)
        if name == 'analysis':c.add_argument('--include-ml',action='store_true',help='Run the unchanged fixed-parameter 33-fold supplemental ML only when explicitly selected.')
    a = p.parse_args()
    input_root = a.input_root.resolve()
    if a.command == 'check-inputs':
        report = check_inputs(a.mode,input_root);print(json.dumps(report,indent=2));return 0 if report['ready'] else 2
    work = a.work_dir.resolve()
    run_analysis(input_root,work,a.include_ml)
    return 0


if __name__ == '__main__':
    sys.exit(main_cli())
