"""Invented arrays only. These are software checks, not study-result evidence."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/production'))
from run_worldmove_five_city_pilot import infer_anchor, exposure_arrays, restricted_anchor


class PairedLogic(unittest.TestCase):
    def test_complete_pair_excludes_missing_mobile_or_anchor(self):
        t=np.zeros((2,48),dtype=np.int64);t[1,12:]=1
        a=infer_anchor(t,np.arange(12));o=np.full((2,24),10,dtype=np.float32)
        o[1,8]=np.nan
        r=exposure_arrays(t,a,o)
        self.assertEqual(r.complete.tolist(),[True,False])
        self.assertTrue(np.isnan(r.delta[1]))

    def test_two_half_hours_share_one_hour(self):
        t=np.zeros((1,48),dtype=np.int64);t[0,16]=1;t[0,17]=2
        o=np.zeros((3,24),dtype=np.float32);o[1,8]=48;o[2,8]=96
        a=infer_anchor(t,np.arange(12));r=exposure_arrays(t,a,o)
        self.assertEqual(float(r.dynamic[0]),3.0)
        self.assertEqual(float(r.diurnal_delta[0,8]),72.0)

    def test_high_confidence_is_subset_same_anchor(self):
        t=np.zeros((2,48),dtype=np.int64);t[0,7:12]=1;t[1,9:12]=1
        a=infer_anchor(t,np.arange(12));h=restricted_anchor(a,a.valid&(a.dominance>=.75))
        self.assertEqual(a.valid.tolist(),[True,True])
        self.assertEqual(h.valid.tolist(),[False,True])
        self.assertTrue(np.array_equal(a.grid_index,h.grid_index))


if __name__=='__main__':unittest.main()
