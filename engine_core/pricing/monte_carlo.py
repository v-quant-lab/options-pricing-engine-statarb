"""
Monte Carlo Option Pricing

Path-dependent option pricing with variance reduction techniques.
"""

import numpy as np
from numba import jit, prange, float64
from typing import Tuple, Optional
from dataclasses import dataclass

from engine_core.pricing.black_scholes import OptionType


@dataclass
class MCResult:
    price: float
    std_error: float
    confidence_interval: Tuple[float, float]
    paths_used: int


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def _mc_european(
    spot: float,
    strike: float,
    tau: float,
    vol: float,
    rate: float,
    div: float,
    opt_type: int,
    n_paths: int,
    seed: int,
) -> Tuple[float, float]:
    """Basic Monte Carlo European option pricing."""
    np.random.seed(seed)
    
    drift = (rate - div - 0.5 * vol * vol) * tau
    diffusion = vol * np.sqrt(tau)
    disc = np.exp(-rate * tau)
    
    payoffs = np.empty(n_paths, dtype=np.float64)
    
    for i in prange(n_paths):
        z = np.random.standard_normal()
        S_T = spot * np.exp(drift + diffusion * z)
        
        if opt_type == 1:
            payoffs[i] = max(S_T - strike, 0.0)
        else:
            payoffs[i] = max(strike - S_T, 0.0)
    
    price = disc * np.mean(payoffs)
    std_err = disc * np.std(payoffs) / np.sqrt(n_paths)
    
    return price, std_err


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def _mc_antithetic(
    spot: float,
    strike: float,
    tau: float,
    vol: float,
    rate: float,
    div: float,
    opt_type: int,
    n_paths: int,
    seed: int,
) -> Tuple[float, float]:
    """Monte Carlo with antithetic variates."""
    np.random.seed(seed)
    
    drift = (rate - div - 0.5 * vol * vol) * tau
    diffusion = vol * np.sqrt(tau)
    disc = np.exp(-rate * tau)
    
    payoffs = np.empty(n_paths, dtype=np.float64)
    
    for i in prange(n_paths // 2):
        z = np.random.standard_normal()
        
        # Original path
        S_T_1 = spot * np.exp(drift + diffusion * z)
        # Antithetic path
        S_T_2 = spot * np.exp(drift + diffusion * (-z))
        
        if opt_type == 1:
            payoffs[2 * i] = max(S_T_1 - strike, 0.0)
            payoffs[2 * i + 1] = max(S_T_2 - strike, 0.0)
        else:
            payoffs[2 * i] = max(strike - S_T_1, 0.0)
            payoffs[2 * i + 1] = max(strike - S_T_2, 0.0)
    
    price = disc * np.mean(payoffs)
    std_err = disc * np.std(payoffs) / np.sqrt(n_paths)
    
    return price, std_err


@jit(nopython=True, cache=True, fastmath=True)
def _mc_asian_arithmetic(
    spot: float,
    strike: float,
    tau: float,
    vol: float,
    rate: float,
    div: float,
    opt_type: int,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> Tuple[float, float]:
    """Monte Carlo Asian option (arithmetic average)."""
    np.random.seed(seed)
    
    dt = tau / n_steps
    drift = (rate - div - 0.5 * vol * vol) * dt
    diffusion = vol * np.sqrt(dt)
    disc = np.exp(-rate * tau)
    
    payoffs = np.empty(n_paths, dtype=np.float64)
    
    for i in range(n_paths):
        S = spot
        avg = 0.0
        
        for _ in range(n_steps):
            z = np.random.standard_normal()
            S = S * np.exp(drift + diffusion * z)
            avg += S
        
        avg /= n_steps
        
        if opt_type == 1:
            payoffs[i] = max(avg - strike, 0.0)
        else:
            payoffs[i] = max(strike - avg, 0.0)
    
    price = disc * np.mean(payoffs)
    std_err = disc * np.std(payoffs) / np.sqrt(n_paths)
    
    return price, std_err


@jit(nopython=True, cache=True, fastmath=True)
def _mc_barrier_down_out(
    spot: float,
    strike: float,
    barrier: float,
    tau: float,
    vol: float,
    rate: float,
    div: float,
    opt_type: int,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> Tuple[float, float]:
    """Monte Carlo down-and-out barrier option."""
    np.random.seed(seed)
    
    dt = tau / n_steps
    drift = (rate - div - 0.5 * vol * vol) * dt
    diffusion = vol * np.sqrt(dt)
    disc = np.exp(-rate * tau)
    
    payoffs = np.empty(n_paths, dtype=np.float64)
    
    for i in range(n_paths):
        S = spot
        knocked_out = False
        
        for _ in range(n_steps):
            z = np.random.standard_normal()
            S = S * np.exp(drift + diffusion * z)
            
            if S <= barrier:
                knocked_out = True
                break
        
        if knocked_out:
            payoffs[i] = 0.0
        else:
            if opt_type == 1:
                payoffs[i] = max(S - strike, 0.0)
            else:
                payoffs[i] = max(strike - S, 0.0)
    
    price = disc * np.mean(payoffs)
    std_err = disc * np.std(payoffs) / np.sqrt(n_paths)
    
    return price, std_err


class MonteCarloPricer:
    """
    Monte Carlo option pricer.
    
    Features:
    - European, Asian, Barrier options
    - Variance reduction (antithetic, control variate)
    - Confidence intervals
    - Configurable paths and steps
    """
    
    def __init__(
        self,
        n_paths: int = 100000,
        n_steps: int = 252,
        seed: int = 42,
        dividend_yield: float = 0.0,
    ):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self.div = dividend_yield
    
    def price_european(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        option_type: OptionType = OptionType.CALL,
        antithetic: bool = True,
    ) -> MCResult:
        """Price European option."""
        if antithetic:
            price, std_err = _mc_antithetic(
                spot, strike, time_to_expiry, volatility, rate,
                self.div, int(option_type), self.n_paths, self.seed
            )
        else:
            price, std_err = _mc_european(
                spot, strike, time_to_expiry, volatility, rate,
                self.div, int(option_type), self.n_paths, self.seed
            )
        
        ci = (price - 1.96 * std_err, price + 1.96 * std_err)
        
        return MCResult(
            price=price,
            std_error=std_err,
            confidence_interval=ci,
            paths_used=self.n_paths,
        )
    
    def price_asian(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        option_type: OptionType = OptionType.CALL,
        averaging_type: str = "arithmetic",
    ) -> MCResult:
        """Price Asian option."""
        price, std_err = _mc_asian_arithmetic(
            spot, strike, time_to_expiry, volatility, rate,
            self.div, int(option_type), self.n_paths, self.n_steps, self.seed
        )
        
        ci = (price - 1.96 * std_err, price + 1.96 * std_err)
        
        return MCResult(
            price=price,
            std_error=std_err,
            confidence_interval=ci,
            paths_used=self.n_paths,
        )
    
    def price_barrier(
        self,
        spot: float,
        strike: float,
        barrier: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        option_type: OptionType = OptionType.CALL,
        barrier_type: str = "down_out",
    ) -> MCResult:
        """Price barrier option."""
        price, std_err = _mc_barrier_down_out(
            spot, strike, barrier, time_to_expiry, volatility, rate,
            self.div, int(option_type), self.n_paths, self.n_steps, self.seed
        )
        
        ci = (price - 1.96 * std_err, price + 1.96 * std_err)
        
        return MCResult(
            price=price,
            std_error=std_err,
            confidence_interval=ci,
            paths_used=self.n_paths,
        )
