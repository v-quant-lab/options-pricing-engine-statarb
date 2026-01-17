"""
Implied Volatility Solvers

High-performance IV solving with multiple algorithms.
"""

import numpy as np
from numba import jit, float64
from typing import Optional, Literal
from dataclasses import dataclass

from engine_core.pricing.black_scholes import _bs_price, _norm_cdf, _norm_pdf, OptionType


@dataclass
class IVResult:
    iv: float
    converged: bool
    iterations: int
    error: float


@jit(float64(float64, float64, float64, float64, float64), nopython=True, cache=True, fastmath=True)
def _vega(spot: float, strike: float, tau: float, vol: float, rate: float) -> float:
    """Compute vega for Newton-Raphson."""
    if tau <= 1e-10 or vol <= 1e-10:
        return 0.0
    
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol * vol) * tau) / (vol * sqrt_tau)
    
    return spot * _norm_pdf(d1) * sqrt_tau


@jit(nopython=True, cache=True, fastmath=True)
def _newton_raphson_iv(
    target_price: float,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    opt_type: int,
    initial_vol: float,
    max_iter: int,
    tol: float,
) -> tuple:
    """Newton-Raphson IV solver."""
    vol = initial_vol
    
    for i in range(max_iter):
        price = _bs_price(spot, strike, tau, vol, rate, 0.0, opt_type)
        diff = price - target_price
        
        if abs(diff) < tol:
            return vol, True, i + 1, diff
        
        vega = _vega(spot, strike, tau, vol, rate)
        
        if abs(vega) < 1e-10:
            break
        
        vol = vol - diff / vega
        vol = max(0.001, min(5.0, vol))
    
    return vol, False, max_iter, diff


@jit(nopython=True, cache=True, fastmath=True)
def _brent_iv(
    target_price: float,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    opt_type: int,
    vol_low: float,
    vol_high: float,
    max_iter: int,
    tol: float,
) -> tuple:
    """Brent's method IV solver."""
    a, b = vol_low, vol_high
    fa = _bs_price(spot, strike, tau, a, rate, 0.0, opt_type) - target_price
    fb = _bs_price(spot, strike, tau, b, rate, 0.0, opt_type) - target_price
    
    if fa * fb > 0:
        return 0.5 * (a + b), False, 0, abs(fa)
    
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    
    c, fc = a, fa
    d = b - a
    e = d
    
    for i in range(max_iter):
        if abs(fb) < tol:
            return b, True, i + 1, fb
        
        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (a * fb * fc / ((fa - fb) * (fa - fc)) +
                 b * fa * fc / ((fb - fa) * (fb - fc)) +
                 c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)
        
        # Conditions for bisection
        if not (((3 * a + b) / 4 < s < b) or (b < s < (3 * a + b) / 4)):
            s = (a + b) / 2
        elif abs(s - b) >= abs(b - c) / 2:
            s = (a + b) / 2
        elif abs(b - c) < tol:
            s = (a + b) / 2
        
        fs = _bs_price(spot, strike, tau, s, rate, 0.0, opt_type) - target_price
        
        c, fc = b, fb
        
        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs
        
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
    
    return b, False, max_iter, fb


@jit(nopython=True, cache=True, fastmath=True)
def _rational_approx_iv(
    target_price: float,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    opt_type: int,
) -> float:
    """Rational approximation for initial IV guess (simplified Jäckel)."""
    forward = spot * np.exp(rate * tau)
    
    if opt_type == 1:
        intrinsic = max(forward - strike, 0.0)
        normalized_price = target_price * np.exp(rate * tau) / forward
    else:
        intrinsic = max(strike - forward, 0.0)
        normalized_price = target_price * np.exp(rate * tau) / forward
    
    # Brenner-Subrahmanyam approximation
    if abs(forward - strike) < 0.01 * forward:
        # ATM approximation
        return target_price * np.sqrt(2 * np.pi / tau) / spot
    
    moneyness = np.log(forward / strike)
    
    # Simple approximation
    vol_approx = np.sqrt(2 * abs(moneyness) / tau)
    vol_approx = max(0.05, min(2.0, vol_approx))
    
    return vol_approx


@jit(nopython=True, cache=True, fastmath=True, parallel=False)
def _batch_newton_iv(
    target_prices: np.ndarray,
    spot: float,
    strikes: np.ndarray,
    tau: float,
    rate: float,
    opt_types: np.ndarray,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Batch IV solving."""
    n = len(target_prices)
    ivs = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        initial = _rational_approx_iv(
            target_prices[i], spot, strikes[i], tau, rate, int(opt_types[i])
        )
        
        vol, converged, _, _ = _newton_raphson_iv(
            target_prices[i], spot, strikes[i], tau, rate,
            int(opt_types[i]), initial, max_iter, tol
        )
        
        ivs[i] = vol if converged else np.nan
    
    return ivs


class ImpliedVolSolver:
    """
    High-performance implied volatility solver.
    
    Features:
    - Multiple algorithms (Newton-Raphson, Brent, Hybrid)
    - Rational approximation for initial guess
    - Batch solving
    - Configurable tolerance
    """
    
    def __init__(
        self,
        method: Literal["newton", "brent", "hybrid"] = "hybrid",
        max_iterations: int = 100,
        tolerance: float = 1e-8,
        bounds: tuple = (0.001, 5.0),
    ):
        self.method = method
        self.max_iter = max_iterations
        self.tol = tolerance
        self.bounds = bounds
        
        # Warm up JIT
        _ = _newton_raphson_iv(5.0, 100.0, 100.0, 0.25, 0.05, 1, 0.2, 10, 1e-6)
    
    def solve(
        self,
        market_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        rate: float,
        option_type: OptionType = OptionType.CALL,
        initial_guess: Optional[float] = None,
    ) -> IVResult:
        """Solve for implied volatility."""
        opt_type = int(option_type)
        
        # Get initial guess
        if initial_guess is None:
            initial_guess = _rational_approx_iv(
                market_price, spot, strike, time_to_expiry, rate, opt_type
            )
        
        if self.method == "newton":
            iv, converged, iters, error = _newton_raphson_iv(
                market_price, spot, strike, time_to_expiry, rate,
                opt_type, initial_guess, self.max_iter, self.tol
            )
        elif self.method == "brent":
            iv, converged, iters, error = _brent_iv(
                market_price, spot, strike, time_to_expiry, rate,
                opt_type, self.bounds[0], self.bounds[1], self.max_iter, self.tol
            )
        else:  # hybrid
            iv, converged, iters, error = _newton_raphson_iv(
                market_price, spot, strike, time_to_expiry, rate,
                opt_type, initial_guess, self.max_iter // 2, self.tol
            )
            
            if not converged:
                iv, converged, iters2, error = _brent_iv(
                    market_price, spot, strike, time_to_expiry, rate,
                    opt_type, self.bounds[0], self.bounds[1], self.max_iter // 2, self.tol
                )
                iters += iters2
        
        return IVResult(iv=iv, converged=converged, iterations=iters, error=abs(error))
    
    def solve_batch(
        self,
        market_prices: np.ndarray,
        spot: float,
        strikes: np.ndarray,
        time_to_expiry: float,
        rate: float,
        option_types: np.ndarray,
    ) -> np.ndarray:
        """Batch IV solving."""
        return _batch_newton_iv(
            market_prices, spot, strikes, time_to_expiry, rate,
            option_types, self.max_iter, self.tol
        )
    
    def solve_chain(
        self,
        chain_prices: np.ndarray,
        spot: float,
        strikes: np.ndarray,
        expiries: np.ndarray,
        rate: float,
        option_types: np.ndarray,
    ) -> np.ndarray:
        """Solve IV for full option chain."""
        n = len(chain_prices)
        ivs = np.empty(n)
        
        for i in range(n):
            result = self.solve(
                chain_prices[i], spot, strikes[i], expiries[i],
                rate, OptionType(option_types[i])
            )
            ivs[i] = result.iv if result.converged else np.nan
        
        return ivs
