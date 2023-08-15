import numpy as np

def rmse(pred, true):
    """Calculates root mean square error

    Args:
        pred (array): Forecasted/predicted values
        true (array): True/observed values

    Returns:
        float: Root mean square error of the two arrays
    """
    return np.sqrt(np.nanmean((np.array(pred) - np.array(true)) ** 2))

def mse(pred, true):
    """Calculates mean square error

    Args:
        pred (array): Forecasted/predicted values
        true (array): True/observed values

    Returns:
        float: Mean square error of the two arrays
    """
    return np.nanmean((np.array(true) - np.array(pred)) ** 2)

def sse(pred, true):
    """Calculates sum of squared errors (residual sum of squares)

    Args:
        pred (array): Forecasted/predicted values
        true (array): True/observed values

    Returns:
        float: Sum of squared errors of the two arrays
    """
    return np.nansum((np.array(true)-np.array(pred)) ** 2)

def mae(pred, true):
    """Calculates mean absolute error

    Args:
        pred (array): Forecasted/predicted values
        true (array): True/observed values

    Returns:
        float: Mean absolute error of the two arrays
    """
    return np.nanmean(np.abs(np.array(true)-np.array(pred)))

def mape(pred, true):
    """Calculates mean absolute percentage error

    Args:
        pred (array): Forecasted/predicted values
        true (array): True/observed values

    Returns:
        float: Mean absolute percentage error of the two arrays
    """   
    return np.nanmean(np.abs(np.array(true)-np.array(pred)/np.array(true)))


def smape(pred, true):
    """Calculates symmetric mean absolute percentage error

    Args:
        pred (array): Forecasted/predicted values
        true (array): True/observed values

    Returns:
        float: Symmetric mean absolute percentage error of the two arrays
    """
    pred = np.array(pred)
    true = np.array(true)
    
    # Find the mean for each pair of pred and true values
    mean_denominator = np.nanmean(np.vstack((pred, true)), axis=0)
    smape_values = np.abs(pred - true) / mean_denominator
    
    return np.nanmean(smape_values)

def lqe(pred, true):
    """Calculates lqe

    Args:
        pred (array): Forecasted/predicted values
        true (array): True/observed values

    Returns:
        float: lqe of the two arrays
    """
    return np.nansum(np.log(np.array(pred)) - np.log(np.array(true)))

def theil_stats_u1(pred, true):
    """Calculates Theil's U1

    Args:
        pred (array): Forecasted/predicted values
        true (array): True/observed values

    Returns:
        float: Theil's U1 of the two arrays
    """
    pred = np.array(pred)
    true = np.array(true)

    rmse = np.sqrt(np.nanmean((pred - true)**2))
    x_magnitude = np.sqrt(np.nanmean(true**2))
    y_magnitude = np.sqrt(np.nanmean(true**2))
    
    return rmse/(x_magnitude + y_magnitude)

def theil_stats_u2(pred, true):
    """Calculates Theil's U2

    Args:
        pred (array): Forecasted/predicted values
        true (array): True/observed values

    Returns:
        float: Theil's U2 of the two arrays
    """
    pred = np.array(pred)
    true = np.array(true)
    
    rmse = np.sqrt(np.nanmean((pred - true)**2))
    y_magnitude = np.sqrt(np.nanmean(true**2))
    
    return rmse/y_magnitude

def theil_stats_dbias(pred, true):
    """Calculates Theil's dbias

    Args:
        pred (array): Forecasted/predicted values
        true (array): True/observed values

    Returns:
        float: Theil's dbias of the two arrays
    """
    pred = np.array(pred)
    true = np.array(true)
    
    return (np.nanmean(pred) - np.nanmean(true))**2 / np.nanmean((pred - true)**2)

def theil_stats_dvar(pred, true):
    """Calculates Theil's dvar

    Args:
        pred (array): Forecasted/predicted values
        true (array): True/observed values

    Returns:
        float: Theil's dvar of the two arrays
    """
    # R calculates standard deviation differently from Python so we have "ddof=1" in the Python function
    pred = np.array(pred)
    true = np.array(true)
    
    return (np.nanstd(pred, ddof=1) - np.nanstd(true, ddof=1))**2 / np.nanmean((pred - true)**2)

def theil_stats_dnoise(pred, true):
    """Calculates Theil's dnoise

    Args:
        pred (array): Forecasted/predicted values
        true (array): True/observed values

    Returns:
        float: Theil's dnoise of the two arrays
    """
    return 1 - theil_stats_dbias(pred, true) - theil_stats_dvar(pred, true)

def theil_stats_stlcomp(pred, true, ts):
    """Calculates Theil's stlcomp

    Args:
        pred (array): Forecasted/predicted values
        true (array): True/observed values
        ts (array): trend and seasonal values

    Returns:
        float: Theil's stlcomp of the two arrays
    """
    return rmse(pred, true) / rmse(true, ts)

def theil_stats(type, x, y, ts):
    """Calculates all Theil's stats

    Args:
        type (array or str): Specify what kind of Theil's stats you want. Please a list with "u1", "u2", "bias", "var", "noise", "stlcomp", as the options or use "all" if you want all stats.
        pred (array): Forecasted/predicted values
        true (array): True/observed values
        ts (array): trend and seasonal values

    Returns:
        dict or float: Theil stats
    """
    if isinstance(type, str):
        if type not in ['u1', 'u2', 'bias', 'var', 'noise', 'stlcomp', 'all']:
            return f'Error - Theil Statistic, "{type}",  not matched. Please use options u1, u2, bias, var, noise, stlcomp, or all'
    elif not set(type).issubset(['u1', 'u2', 'bias', 'var', 'noise', 'stlcomp', 'all']):
        return f'Error - Theil Statistic, "{type}",  not matched. Please use options u1, u2, bias, var, noise, stlcomp, or all'

    theil_stats = {
            'u1':
            theil_stats_u1(x, y),
            'u2':
            theil_stats_u2(x, y),
            'bias':
            theil_stats_dbias(x, y),
            'var':
            theil_stats_dvar(x, y),
            'noise':
            theil_stats_dnoise(x, y),
            'stlcomp':
            theil_stats_stlcomp(x, y, ts)
    }

    if 'all' in type:
        return theil_stats
    else:
        return {key: theil_stats[key] for key in type}
    