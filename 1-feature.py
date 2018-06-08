#!/bin/env python

from msmbuilder.io import gather_metadata, save_meta, NumberedRunsParser
import numpy as np
import mdtraj as md
from msmbuilder.featurizer import DihedralFeaturizer, AtomPairsFeaturizer
from msmbuilder.io import load_meta, preload_tops, save_trajs, save_generic
from multiprocessing import Pool
import contextlib

meta = load_meta("meta.pandas.pickl")
tops = preload_tops(meta)

alpha_carbon_number = np.array([9,26,40,59,80,92,104,111,118,139,151,170,187,194,215,234,253,270,289,306,320,346,358,374,386,403,419,434,453,462,474,490,502,516,527,538,548,567,586,605,616,628,647,669,686,708,720,736,748,767,783,804,814,825,840,850,870,889,910,927,941,948,969,980,994,1004,1019,1035,1054,1061,1085,1099,1109,1133,1153,1172,1189,1202,1214,1226,1233,1250,1266,1290,1302,1324,1335,1349,1373,1395,1416,1432,1444,1455,1469,1483,1502,1516,1530,1547,1571,1593,1603,1622,1634,1658,1672,1682,1697,1713,1730,1746,1761,1777,1793,1807,1827,1849,1871,1885,1892,1909,1933,1953,1969,1983,2003,2022,2036,2053,2074,2086,2102,2126,2138,2153,2167,2174,2189,2210,2234,2255,2266,2283,2290,2310,2327,2338])
num=len(alpha_carbon_number)

atompair=[]
for i in range(num):
    for j in range(i+1,num):
        atompair += [[alpha_carbon_number[i],alpha_carbon_number[j]]]
dist_feat = AtomPairsFeaturizer(pair_indices=atompair)  ## Distance featurizer

def feat2(irow):
    i, row = irow
    traj = md.load(row['traj_fn'], top=tops[row['top_fn']])
    feat_traj = dist_feat.partial_transform(traj)
    return i, feat_traj

with contextlib.closing(Pool(processes=32)) as pool:
    dist_trajs = dict(pool.imap_unordered(feat2, meta.iterrows()))
    
save_trajs(dist_trajs, 'alpha_carbon', meta)

