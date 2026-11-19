"""Train the comparison models and load transformer checkpoints."""
import json
from pathlib import Path

import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import PixelDataset, collate
from .models.transformer import AgriMatNetUAV, quantile_loss
from .models.linear_regression import fit
from .utils import seed_all, device_batch, relocate_checkpoint_metadata


def loader(dataset, cfg, shuffle=False):
    generator = torch.Generator().manual_seed(cfg['seed'])
    return DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=shuffle,
                      collate_fn=collate, num_workers=cfg['num_workers'], generator=generator)


def train(examples, out, cfg):
    seed_all(cfg['seed'])
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh run directory to preserve existing checkpoints')
    out.mkdir(parents=True, exist_ok=True)
    train_set, val_set = PixelDataset(examples, 'train'), PixelDataset(examples, 'validation')
    if cfg.get('experiment', 'spatial_generalization') != train_set.cfg.get('experiment', 'spatial_generalization'):
        raise ValueError('Experiment differs from examples')
    for key in ['excluded_dates', 'train_years', 'test_year']:
        if cfg.get(key) != train_set.cfg.get(key):
            raise ValueError(f'{key} differs from example configuration')
    if cfg.get('validation_dates') and cfg['validation_dates'] != train_set.cfg.get('validation_dates'):
        raise ValueError('Validation dates differ from examples')
    cfg = dict(cfg, validation_dates=train_set.cfg.get('validation_dates', []))
    # Features and sampling semantics must match the persisted example set.
    for key in ['threshold', 'min_history', 'max_history', 'max_history_days', 'max_horizon_days',
                'extra_features', 'split_axis', 'buffer_pixels']:
        if cfg[key] != train_set.cfg[key]:
            raise ValueError(f'{key} differs from example configuration; rebuild examples')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AgriMatNetUAV(cfg, 1+len(cfg['extra_features'])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    train_loader, val_loader = loader(train_set, cfg, True), loader(val_set, cfg)
    best, stale, records = float('inf'), 0, []
    median = cfg['quantiles'].index(.5)
    for epoch in range(cfg['epochs']):
        stats = {}
        for name, batches in [('train', train_loader), ('validation', val_loader)]:
            model.train(name == 'train')
            losses, squared, count = 0., 0., 0
            with torch.set_grad_enabled(name == 'train'):
                for batch in batches:
                    batch = device_batch(batch, device)
                    pred = model(batch)
                    loss = quantile_loss(pred, batch['target'], cfg['quantiles'])
                    if cfg.get('loss', 'pinball') == 'mse':
                        loss = ((pred[:, median]-batch['target'])**2).mean() + .1*loss
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss')
                    if name == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    n = len(pred)
                    losses += loss.item()*n
                    squared += ((pred[:, median]-batch['target'])**2).sum().item()
                    count += n
            stats[f'{name}_loss'] = losses/count
            stats[f'{name}_rmse'] = (squared/count)**.5
        records.append(dict(epoch=epoch+1, **stats))
        print(f"Epoch {epoch+1}: train RMSE={stats['train_rmse']:.5f}, validation RMSE={stats['validation_rmse']:.5f}", flush=True)
        if stats['validation_rmse'] < best:
            best, stale = stats['validation_rmse'], 0
            torch.save(dict(state_dict=model.state_dict(), config=cfg, metadata=train_set.meta,
                            epoch=epoch+1, validation_rmse=best), out / 'transformer.pt')
        else:
            stale += 1
        pd.DataFrame(records).to_csv(out / 'history.csv', index=False)
        if stale >= cfg['patience']:
            break
    joblib.dump(fit(train_set), out / 'linear.joblib')
    (out / 'run.json').write_text(json.dumps(dict(config=cfg, examples=str(Path(examples).resolve()),
         device=str(device), best_validation_rmse=best), indent=2))
    from .evaluate import plot_history
    plot_history(out)
    return records


def load_model(run, device='cpu'):
    saved = torch.load(Path(run) / 'transformer.pt', map_location=device, weights_only=False)
    saved['metadata'] = relocate_checkpoint_metadata(saved['metadata'])
    cfg = saved['config']
    model = AgriMatNetUAV(cfg, 1+len(cfg['extra_features'])).to(device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model, saved
