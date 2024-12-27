import numpy as np
from scipy.stats import ttest_rel

def compare_metrics(metric_a, metric_b):
    """
    Performs a paired t-test between two arrays of metrics.
    Returns the t-statistic and p-value.
    """
    t_stat, p_val = ttest_rel(metric_a, metric_b)
    return t_stat, p_val

def cross_val_metrics(model_func, X, y, cv=5, **kwargs):
    """
    utility to perform multiple train/validation splits (k-fold style),
    capture metrics for each fold, and return an array of results.
    """
    metrics_list = []
    indices = np.array_split(range(len(X)), cv)
    for i in range(cv):
        val_idx = indices[i]
        train_idx = np.concatenate([indices[j] for j in range(cv) if j != i])
        
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
    
        model = model_func(X_train, y_train, test_size=0, **kwargs)
        if isinstance(model, tuple) and len(model) == 3:
            clf, acc, auc = model
            metrics_list.append((acc, auc))
    
    return metrics_list
