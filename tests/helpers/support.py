"""Fixtures based on the supplied contract; never inspect implementation source."""
import os
import sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/beamforming-verification-mpl')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from helpers.cellular_topology import CellularTopology
from helpers.electrical_downtilt import ElectricalDowntilt
from helpers.kpi_manager import KpiManager
from helpers.large_scale_channel import LargeScaleState
from sionna.phy.channel.tr38901 import AntennaArray

def array(rows=8, cols=1, polarization='single'):
    return AntennaArray(rows, cols, polarization, 'V' if polarization == 'single' else 'VH',
                        'omni', 3.5e9, device='cpu')

def topology(num_sites=None, batch_size=1):
    params = {k: torch.tensor(v) for k, v in
              dict(isd=500., bs_height=25., min_bs_ut_dist=35., min_ut_height=1.5).items()}
    return CellularTopology(params, 1, num_sites=num_sites, batch_size=batch_size, device='cpu')

def manager():
    adjacency = ~np.eye(3, dtype=bool)
    return KpiManager([ElectricalDowntilt(array(), 3.5e9) for _ in range(3)],
                      10., 1e-10, np.zeros((3, 2)),
                      np.array([[1, 2], [0, 2], [0, 1]]), 2, adjacency)

def state(batch=2, users=5):
    shape = (batch, 3, users)
    return LargeScaleState(torch.full(shape, 100.),
                           torch.linspace(1.1, 1.6, int(np.prod(shape))).reshape(shape),
                           torch.zeros(shape))

def numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
