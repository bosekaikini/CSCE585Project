import numpy as np
from src.baselines import black_scholes_call, black_scholes_delta, black_scholes_gamma

def test_black_scholes_call_closed_form():
    assert np.isclose(black_scholes_call(100,100,1,.05,.2), 10.4505835722, rtol=1e-9)

def test_black_scholes_analytic_delta_gamma():
    assert np.isclose(black_scholes_delta(100,100,1,.05,.2), .6368306512, rtol=1e-8)
    assert np.isclose(black_scholes_gamma(100,100,1,.05,.2), .0187620173, rtol=1e-8)
