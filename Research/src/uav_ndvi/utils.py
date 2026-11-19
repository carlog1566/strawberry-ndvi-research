"""Shared reproducibility and tensor-device helpers."""
import json
from pathlib import Path
import random

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def relocate_checkpoint_metadata(metadata):
    """Resolve embedded workspace paths without modifying saved model binaries."""
    project = Path(__file__).resolve().parents[2]
    marker = project / '.relocation.json'
    if not marker.is_file():
        return metadata
    roots = json.loads(marker.read_text(encoding='utf-8'))['previous_roots']
    current = str(project).replace('\\', '/')

    def relocate(value):
        if isinstance(value, str):
            normal = value.replace('\\', '/')
            if normal.casefold() == current.casefold() or normal.casefold().startswith(current.casefold() + '/'):
                return value
            for root in sorted(roots, key=len, reverse=True):
                previous = root.replace('\\', '/').rstrip('/')
                if normal.casefold() == previous.casefold() or normal.casefold().startswith(previous.casefold() + '/'):
                    return str(project / normal[len(previous):].lstrip('/'))
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        return value

    return relocate(metadata)


__all__ = ['seed_all', 'device_batch']
