"""Supervised regression features and fitting for the current pipeline."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def features(item):
    x = item['history'].numpy()
    valid = ~item['missing'].numpy()
    y = x[valid, 0].astype('float64')
    t = item['days'].numpy()[valid].astype('float64')
    lead = float(item['target_day'])-t[-1]
    slope = np.polyfit(t-t[0], y, 1)[0] if len(t) > 1 else 0.
    span = t[-1]-t[0]
    result = [y[-1], y[0], y.mean(), y.std(), len(y), span, lead,
              slope, slope*lead, y[-1]*lead, *item['future'].numpy()[1:]]
    # Optional past-only feature means; missing values have separate indicators.
    channels = (x.shape[1]-2)//2
    for j in range(1, channels):
        available = (x[:, channels+j] == 0) & valid
        result.extend([float(x[available, j].mean()) if available.any() else 0., float(available.any())])
    return np.asarray(result, dtype='float64')


def matrix(dataset):
    return np.stack([features(dataset[i]) for i in range(len(dataset))])


def fit(dataset):
    model = make_pipeline(StandardScaler(), LinearRegression())
    y = np.array([float(dataset[i]['target']) for i in range(len(dataset))])
    model.fit(matrix(dataset), y)
    return model


__all__ = ['features', 'matrix', 'fit']
