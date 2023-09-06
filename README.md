# Error Metrics Library

`errormetrics` is a small library that provides different error metrics. These metrics are mainly used in the time series forecasting.

## Get Started

Use `pip install git+https://github.com/jaciz/error_metrics/` to install the library.

## Examples

To get started, import the library as such below.

``` Python
from errormetrics import errormetrics as em

# Example data
x = [1, 2, 3, 4, 5]
y = [2, 5, 6, 4, 9]
ts =  [1, 3, 5, 2, 3]
```

## RMSE

``` Python
In [1]: em.rmse(x, y)

Out[2]: 
2.6457513110645907
```

## MSE

``` Python
In [1]: em.mse(x, y)

Out[2]: 
7.0
```

## SSE

``` Python
In [1]: em.sse(x, y)

Out[2]: 
35
```

## MAE

``` Python
In [1]: em.mae(x, y)

Out[2]: 
2.2
```

## MAPE

``` Python
In [1]: em.mape(x, y)

Out[2]: 
0.40888888888888886
```

## SMAPE

``` Python
In [1]: em.smape(x, y)

Out[2]: 
0.5523809523809524
```

## LQE

``` Python
In [1]: em.lqe(x, y)

Out[2]: 
-2.8903717578961645
```

## Theil's U1

``` Python
In [1]: em.theil_stats_u1(x, y)

Out[2]: 
0.29368766776734756
```

## Theil's U2

``` Python
In [1]: em.theil_stats_u2(x, y)

Out[2]: 
0.4648111258522642
```

## Theil's Stats DBias

``` Python
In [1]: em.theil_stats_dbias(x, y)

Out[2]: 
0.6914285714285715
```

## Theil's Stats DVar

``` Python
In [1]: em.theil_stats_dvar(x, y)

Out[2]: 
0.14494960401822152
```

## Theil's Stats DNoise

``` Python
In [1]: em.theil_stats_dnoise(x, y)

Out[2]: 
0.16362182455320698
```

## Theil's Stats STL Comp

``` Python
In [1]: em.theil_stats_stlcomp(x, y, ts)

Out[2]: 
0.872278375988647
```

## Theil's Stats

``` Python
In [1]: em.theil_stats('all', x, y, ts)

Out[2]: 
{'u1': 0.29368766776734756,
 'u2': 0.4648111258522642,
 'bias': 0.6914285714285715,
 'var': 0.14494960401822152,
 'noise': 0.16362182455320698,
 'stlcomp': 0.872278375988647}
```
