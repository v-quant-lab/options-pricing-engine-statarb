"""
Black-Scholes Pricing Engine with Numba JIT Compilation

High-performance vectorized option pricing.
"""

import numpy as np
from numba import jit, prange, float64, int32, vectorize
from enum import IntEnum
from typing import Union, Tuple
from dataclasses import dataclass


class OptionType(IntEnum):
    CALL = 1
    PUT = -1


@dataclass
class PricingResult:
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


@jit(float64(float64), nopython=True, cache=True, fastmath=True)
def _norm_cdf(x: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1.0 + np.math.erf(x / np.sqrt(2.0)))


@jit(float64(float64), nopython=True, cache=True, fastmath=True)
def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


@jit(
    float64(float64, float64, float64, float64, float64, float64, int32),
    nopython=True,
    cache=True,
    fastmath=True,
)
def _bs_price(
    spot: float,
    strike: float,
    tau: float,
    vol: float,
    rate: float,
    div: float,
    opt_type: int,
) -> float:
    """Black-Scholes price calculation."""
    if tau <= 0:
        if opt_type == 1:  # Call
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)
    
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(spot / strike) + (rate - div + 0.5 * vol * vol) * tau) / (vol * sqrt_tau)
    d2 = d1 - vol * sqrt_tau
    
    df = np.exp(-rate * tau)
    fwd = spot * np.exp((rate - div) * tau)
    
    if opt_type == 1:  # Call
        return df * (fwd * _norm_cdf(d1) - strike * _norm_cdf(d2))
    else:  # Put
        return df * (strike * _norm_cdf(-d2) - fwd * _norm_cdf(-d1))


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def _bs_price_batch(
    spot: float,
    strikes: np.ndarray,
    tau: float,
    vol: float,
    rate: float,
    div: float,
    opt_type: int,
) -> np.ndarray:
    """Vectorized Black-Scholes pricing."""
    n = len(strikes)
    prices = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        prices[i] = _bs_price(spot, strikes[i], tau, vol, rate, div, opt_type)
    
    return prices


@jit(nopython=True, cache=True, fastmath=True)
def _bs_greeks(
    spot: float,
    strike: float,
    tau: float,
    vol: float,
    rate: float,
    div: float,
    opt_type: int,
) -> Tuple[float, float, float, float, float, float]:
    """Compute all Greeks in single pass."""
    if tau <= 1e-10:
        intrinsic = max(opt_type * (spot - strike), 0.0)
        delta = float(opt_type) if intrinsic > 0 else 0.0
        return intrinsic, delta, 0.0, 0.0, 0.0, 0.0
    
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(spot / strike) + (rate - div + 0.5 * vol * vol) * tau) / (vol * sqrt_tau)
    d2 = d1 - vol * sqrt_tau
    
    df = np.exp(-rate * tau)
    df_div = np.exp(-div * tau)
    
    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)
    npd1 = _norm_pdf(d1)
    
    # Price
    if opt_type == 1:
        price = spot * df_div * nd1 - strike * df * nd2
        delta = df_div * nd1
        rho = strike * tau * df * nd2 / 100.0
    else:
        price = strike * df * _norm_cdf(-d2) - spot * df_div * _norm_cdf(-d1)
        delta = -df_div * _norm_cdf(-d1)
        rho = -strike * tau * df * _norm_cdf(-d2) / 100.0
    
    # Gamma (same for call/put)
    gamma = df_div * npd1 / (spot * vol * sqrt_tau)
    
    # Vega (same for call/put)
    vega = spot * df_div * npd1 * sqrt_tau / 100.0
    
    # Theta
    term1 = -spot * df_div * npd1 * vol / (2.0 * sqrt_tau)
    if opt_type == 1:
        term2 = div * spot * df_div * nd1
        term3 = -rate * strike * df * nd2
    else:
        term2 = -div * spot * df_div * _norm_cdf(-d1)
        term3 = rate * strike * df * _norm_cdf(-d2)
    theta = (term1 + term2 + term3) / 365.0
    
    return price, delta, gamma, theta, vega, rho


class BlackScholesPricer:
    """
    High-performance Black-Scholes option pricer.
    
    Features:
    - JIT-compiled core functions
    - Vectorized batch pricing
    - Full Greeks computation
    - Dividend yield support
    """
    
    def __init__(self, dividend_yield: float = 0.0):
        self.div = dividend_yield
        # Warm up JIT compilation
        _ = _bs_price(100.0, 100.0, 0.25, 0.2, 0.05, 0.0, 1)
        _ = _bs_price_batch(100.0, np.array([100.0]), 0.25, 0.2, 0.05, 0.0, 1)
    
    def price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        option_type: OptionType = OptionType.CALL,
        dividend_yield: float = None,
    ) -> float:
        """Price a single option."""
        div = dividend_yield if dividend_yield is not None else self.div
        return _bs_price(
            spot, strike, time_to_expiry, volatility, rate, div, int(option_type)
        )
    
    def price_batch(
        self,
        spot: float,
        strikes: np.ndarray,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        option_type: OptionType = OptionType.CALL,
        dividend_yield: float = None,
    ) -> np.ndarray:
        """Price multiple options with same parameters except strike."""
        div = dividend_yield if dividend_yield is not None else self.div
        strikes = np.ascontiguousarray(strikes, dtype=np.float64)
        return _bs_price_batch(
            spot, strikes, time_to_expiry, volatility, rate, div, int(option_type)
        )
    
    def price_chain(
        self,
        spot: float,
        strikes: np.ndarray,
        expiries: np.ndarray,
        volatilities: np.ndarray,
        rate: float,
        option_types: np.ndarray,
    ) -> np.ndarray:
        """Price full option chain with different parameters."""
        n = len(strikes)
        prices = np.empty(n, dtype=np.float64)
        
        for i in range(n):
            prices[i] = _bs_price(
                spot, strikes[i], expiries[i], volatilities[i],
                rate, self.div, int(option_types[i])
            )
        
        return prices
    
    def greeks(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        option_type: OptionType = OptionType.CALL,
    ) -> PricingResult:
        """Compute price and all Greeks."""
        price, delta, gamma, theta, vega, rho = _bs_greeks(
            spot, strike, time_to_expiry, volatility, rate, self.div, int(option_type)
        )
        return PricingResult(
            price=price,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
        )


class GreeksCalculator:
    """Dedicated Greeks calculator with additional sensitivities."""
    
    def __init__(self, pricer: BlackScholesPricer = None):
        self.pricer = pricer or BlackScholesPricer()
    
    def compute_all(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        option_type: OptionType = OptionType.CALL,
    ) -> PricingResult:
        """Compute all first-order Greeks."""
        return self.pricer.greeks(
            spot, strike, time_to_expiry, volatility, rate, option_type
        )
    
    def delta(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        option_type: OptionType = OptionType.CALL,
    ) -> float:
        """Compute delta only."""
        return self.pricer.greeks(
            spot, strike, time_to_expiry, volatility, rate, option_type
        ).delta
    
    def gamma(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
    ) -> float:
        """Compute gamma (same for call/put)."""
        return self.pricer.greeks(
            spot, strike, time_to_expiry, volatility, rate, OptionType.CALL
        ).gamma
    
    def vanna(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        dS: float = 0.01,
        dV: float = 0.001,
    ) -> float:
        """Compute vanna (d²V/dSdσ) via finite difference."""
        base = self.pricer.greeks(spot, strike, time_to_expiry, volatility, rate, OptionType.CALL)
        up_s = self.pricer.greeks(spot * (1 + dS), strike, time_to_expiry, volatility, rate, OptionType.CALL)
        
        return (up_s.vega - base.vega) / (spot * dS)
    
    def volga(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        dV: float = 0.01,
    ) -> float:
        """Compute volga (d²V/dσ²) via finite difference."""
        up = self.pricer.greeks(spot, strike, time_to_expiry, volatility + dV, rate, OptionType.CALL)
        down = self.pricer.greeks(spot, strike, time_to_expiry, volatility - dV, rate, OptionType.CALL)
        base = self.pricer.greeks(spot, strike, time_to_expiry, volatility, rate, OptionType.CALL)
        
        return (up.vega - 2 * base.vega + down.vega) / (dV * dV)
