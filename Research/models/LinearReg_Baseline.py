#!/usr/bin/env python3
"""Compatibility launcher for the standalone legacy regression workflow."""
from pathlib import Path
import sys

# Keep direct execution working from any directory without installation.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uav_ndvi import legacy_linear_regression as _implementation
from uav_ndvi.legacy_linear_regression import (
    NODATA,
    FEATURE_NAMES,
    _WARNED_NODATA,
    read_manifest,
    build_grid,
    aligned_ndvi,
    valid_input_pixels,
    feature_matrix,
    spatial_region_masks,
    collect_samples,
    show_metrics,
    train,
    predict,
    main,
)

__doc__ = _implementation.__doc__

if __name__ == '__main__':
    main()
