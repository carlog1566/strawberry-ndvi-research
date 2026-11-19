"""Compatibility imports; implementation is in train.py."""
from .train import train, load_model, loader, seed_all, device_batch

__all__ = ['train', 'load_model', 'loader', 'seed_all', 'device_batch']
