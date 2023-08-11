import numpy as np

def rmse(x, y):
    """Calculates root mean square error

    Args:
        x (array): values
        y (array): values

    Returns:
        float: Root mean square error of the two arrays
    """
    return np.sqrt(np.nanmean((np.array(x) - np.array(y)) ** 2))

def mse(x, y):
    """Calculates mean square error

    Args:
        x (array): values
        y (array): values

    Returns:
        float: Mean square error of the two arrays
    """
    return np.nanmean((np.array(x) - np.array(y)) ** 2)

def sse(x, y):
    """Calculates sum of squared errors

    Args:
        x (array): values
        y (array): values

    Returns:
        float: Sum of squared errors of the two arrays
    """
    return np.nansum((np.array(x)-np.array(y)) ** 2)

def mae(x, y):
    """Calculates mean absolute error

    Args:
        x (array): values
        y (array): values

    Returns:
        float: Mean absolute error of the two arrays
    """
    return np.nanmean(np.abs(np.array(x)-np.array(y)))

def mape(x, y):
    """Calculates mean absolute percentage error

    Args:
        x (array): values
        y (array): values

    Returns:
        float: Mean absolute percentage error of the two arrays
    """   
    return np.nanmean(np.abs(np.array(x)-np.array(y))/np.array(y))


def smape(x, y):
    """Calculates symmetric mean absolute percentage error

    Args:
        x (array): values
        y (array): values

    Returns:
        float: Symmetric mean absolute percentage error of the two arrays
    """
    x = np.array(x)
    y = np.array(y)
    
    mean_denominator = np.nanmean(np.vstack((x, y)), axis=0)
    smape_values = np.abs(x - y) / mean_denominator
    
    return np.nanmean(smape_values)

def lqe(x, y):
    """Calculates lqe

    Args:
        x (array): values
        y (array): values

    Returns:
        float: lqe of the two arrays
    """
    return np.nansum(np.log(np.array(x)) - np.log(np.array(y)))

def theil_stats_u1(x, y):
    """Calculates Theil's U1

    Args:
        x (array): values
        y (array): values

    Returns:
        float: Theil's U1 of the two arrays
    """
    x = np.array(x)
    y = np.array(y)

    rmse = np.sqrt(np.nanmean((x - y)**2))
    x_magnitude = np.sqrt(np.nanmean(x**2))
    y_magnitude = np.sqrt(np.nanmean(y**2))
    
    return rmse/(x_magnitude + y_magnitude)

def theil_stats_u2(x, y):
    """Calculates Theil's U2

    Args:
        x (array): values
        y (array): values

    Returns:
        float: Theil's U2 of the two arrays
    """
    x = np.array(x)
    y = np.array(y)
    
    rmse = np.sqrt(np.nanmean((x - y)**2))
    y_magnitude = np.sqrt(np.nanmean(y**2))
    
    return rmse/y_magnitude

def theil_stats_dbias(x, y):
    """Calculates Theil's dbias

    Args:
        x (array): values
        y (array): values

    Returns:
        float: Theil's dbias of the two arrays
    """
    x = np.array(x)
    y = np.array(y)
    
    return (np.nanmean(x) - np.nanmean(y))**2 / np.nanmean((x - y)**2)

def theil_stats_dvar(x, y):
    """Calculates Theil's dvar

    Args:
        x (array): values
        y (array): values

    Returns:
        float: Theil's dvar of the two arrays
    """
    # R calculates standard deviation differently from Python so we have "ddof=1" in the Python function
    x = np.array(x)
    y = np.array(y)
    
    return (np.nanstd(x, ddof=1) - np.nanstd(y, ddof=1))**2 / np.nanmean((x - y)**2)

def theil_stats_dnoise(x, y):
    """Calculates Theil's dnoise

    Args:
        x (array): values
        y (array): values

    Returns:
        float: Theil's dnoise of the two arrays
    """
    return 1 - theil_stats_dbias(x, y) - theil_stats_dvar(x, y)

def theil_stats_stlcomp(x, y, ts):
    """Calculates Theil's stlcomp

    Args:
        x (array): values
        y (array): values
        ts (array): values

    Returns:
        float: Theil's stlcomp of the two arrays
    """
    return rmse(x, y) / rmse(y, ts)

def theil_stats(type, x, y, ts):
    """Calculates all Theil's stats

    Args:
        x (array): values
        y (array): values
        ts (array): values

    Returns:
        dict or float: Theil stats
    """
    if type not in ['u1', 'u2', 'bias', 'var', 'noise', 'stlcomp', 'all']:
        return f'Error - Theil Statistic, "{type}",  not found. Please use options u1, u2, bias, var, noise, stlcomp, or all'

    if type == 'all':
        return {
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
            theil_stats_stlcomp(x, y, ts),
            }
    else:
        return {
            'u1': theil_stats_u1(x, y),
            'u2': theil_stats_u2(x, y),
            'bias': theil_stats_dbias(x, y),
            'var': theil_stats_dvar(x, y),
            'noise': theil_stats_dnoise(x, y),
            'stlcomp': theil_stats_stlcomp(x, y, ts),
        }[type]

    
    