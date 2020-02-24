from .bps import BPS
from .zigzag import ZigZag
from .local_bps import LocalBPS
from .coordinate_sampler import CoordinateSampler
import sys

if sys.platform != 'win32':
    from src.sampling_algorithms.masked_bps.masked_local_bps import MaskedLocalBPS
