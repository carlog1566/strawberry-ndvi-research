"""Season-aware, spatially isolated UAV NDVI forecasting."""
from pathlib import Path

# Root-directory compatibility without installing or changing the existing venv.
__path__.append(str(Path(__file__).resolve().parents[1] / 'src' / 'uav_ndvi'))
