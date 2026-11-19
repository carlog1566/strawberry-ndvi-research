import json
from pathlib import Path

DEFAULT = json.loads((Path(__file__).parent / 'defaults.json').read_text())
DEFAULT.update(experiment='spatial_generalization', train_years=[2022, 2023, 2024],
               test_year=2025, validation_dates=[], require_seasons=False,
               grid_extent='union', loss='mse', residual_prediction=True, excluded_dates=[])


def load_config(path=None):
    cfg = DEFAULT.copy()
    if path is None and Path('config.yaml').is_file():
        path = 'config.yaml'
    if path:
        text = Path(path).read_text()
        # JSON is valid YAML; native YAML requires optional PyYAML.
        try:
            updates = json.loads(text)
        except json.JSONDecodeError:
            import yaml
            updates = yaml.safe_load(text)
        unknown = updates.keys() - cfg.keys()
        if unknown:
            raise ValueError(f'Unknown configuration keys: {unknown}')
        cfg.update(updates)
    if cfg['min_history'] < 1 or cfg['max_history'] < cfg['min_history']:
        raise ValueError('Require 1 <= min_history <= max_history')
    for key in ['tile_size', 'batch_size', 'epochs', 'patience', 'samples_per_split']:
        if cfg[key] < 1:
            raise ValueError(f'{key} must be positive')
    if cfg['split_axis'] not in ['row', 'column'] or cfg['buffer_pixels'] < 0:
        raise ValueError('Invalid spatial split configuration')
    q = cfg['quantiles']
    if sorted(set(q)) != q or 0.5 not in q or not all(0 < x < 1 for x in q):
        raise ValueError('Quantiles must be unique, sorted, in (0,1), and contain 0.5')
    if cfg['d_model'] % 2 or cfg['d_model'] % cfg['num_heads']:
        raise ValueError('d_model must be even and divisible by num_heads')
    if cfg['experiment'] not in ['spatial_generalization', 'cross_year_2025']:
        raise ValueError('Unknown experiment')
    if cfg['grid_extent'] not in ['union', 'reference'] or cfg['loss'] not in ['mse', 'pinball']:
        raise ValueError('Invalid grid extent or loss')
    if cfg['test_year'] in cfg['train_years']:
        raise ValueError('Test year cannot be a training year')
    if cfg['experiment'] == 'cross_year_2025' and (cfg['test_year'] != 2025 or
            not set(cfg['train_years']) <= {2022, 2023, 2024}):
        raise ValueError('Cross-year 2025 permits only 2022-2024 training and 2025 testing')
    if any(not d.startswith('2024-') for d in cfg['validation_dates']):
        raise ValueError('Validation targets must be dates in 2024')
    return cfg
