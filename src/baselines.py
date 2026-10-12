"""Reference Black-Scholes, Monte Carlo, and finite-difference baselines."""
from __future__ import annotations
import math
import numpy as np

def _norm_cdf(x): return .5*(1+np.vectorize(math.erf)(np.asarray(x)/math.sqrt(2)))

def black_scholes_call(spot, strike, maturity, rate, volatility):
    spot = np.asarray(spot, dtype=float)
    if maturity <= 0: return np.maximum(spot-strike, 0.)
    if volatility <= 0: return np.maximum(spot-strike*np.exp(-rate*maturity), 0.)
    d1 = (np.log(spot/strike)+(rate+.5*volatility**2)*maturity)/(volatility*math.sqrt(maturity))
    d2 = d1-volatility*math.sqrt(maturity)
    return spot*_norm_cdf(d1)-strike*math.exp(-rate*maturity)*_norm_cdf(d2)

def black_scholes_delta(spot, strike, maturity, rate, volatility):
    if maturity <= 0: return (np.asarray(spot)>strike).astype(float)
    d1=(np.log(np.asarray(spot)/strike)+(rate+.5*volatility**2)*maturity)/(volatility*math.sqrt(maturity))
    return _norm_cdf(d1)

def black_scholes_gamma(spot, strike, maturity, rate, volatility):
    d1=(np.log(np.asarray(spot)/strike)+(rate+.5*volatility**2)*maturity)/(volatility*math.sqrt(maturity))
    return np.exp(-d1**2/2)/(np.asarray(spot)*volatility*math.sqrt(2*math.pi*maturity))

def monte_carlo_call(spot, strike, maturity, rate, volatility, paths=100000, seed=None):
    rng=np.random.default_rng(seed); z=rng.standard_normal(paths)
    st=spot*np.exp((rate-.5*volatility**2)*maturity+volatility*math.sqrt(maturity)*z)
    return float(math.exp(-rate*maturity)*np.maximum(st-strike,0).mean())

def finite_difference_call(spot, strike, maturity, rate, volatility, bump=1e-4):
    p=black_scholes_call(spot,strike,maturity,rate,volatility)
    u=black_scholes_call(np.asarray(spot)+bump,strike,maturity,rate,volatility)
    d=black_scholes_call(np.asarray(spot)-bump,strike,maturity,rate,volatility)
    return {"price":p,"delta":(u-d)/(2*bump),"gamma":(u-2*p+d)/bump**2}


def longstaff_schwartz_put(spot, strike, maturity, rate, volatility,
                           paths=10000, steps=50, seed=None):
    """Estimate an American put with Longstaff-Schwartz least-squares exercise.

    Returns the discounted price only; a real experiment should record the
    path count and standard error alongside this estimate.
    """
    rng = np.random.default_rng(seed)
    dt = maturity / steps
    prices = np.empty((paths, steps + 1))
    prices[:, 0] = spot
    shocks = rng.standard_normal((paths, steps))
    for step in range(1, steps + 1):
        prices[:, step] = prices[:, step - 1] * np.exp(
            (rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks[:, step - 1]
        )
    cashflow = np.maximum(strike - prices[:, -1], 0.0)
    for step in range(steps - 1, 0, -1):
        in_money = prices[:, step] < strike
        if in_money.sum() < 3:
            cashflow *= np.exp(-rate * dt)
            continue
        x = prices[in_money, step]
        continuation = np.polyval(np.polyfit(x, cashflow[in_money] * np.exp(-rate * dt), 2), x)
        exercise = np.maximum(strike - x, 0.0)
        cashflow[in_money] = np.where(exercise > continuation, exercise,
                                      cashflow[in_money] * np.exp(-rate * dt))
        cashflow[~in_money] *= np.exp(-rate * dt)
    return float(np.mean(cashflow) * np.exp(-rate * dt))
