"""
Binomial Tree Option Pricing

Cox-Ross-Rubinstein model for American options with early exercise.
"""

import numpy as np
from numba import jit, prange, float64, int32
from typing import Tuple
from dataclasses import dataclass

from engine_core.pricing.black_scholes import OptionType


@dataclass
class TreeResult:
    price: float
    delta: float
    gamma: float
    early_exercise_boundary: np.ndarray


@jit(nopython=True, cache=True, fastmath=True)
def _crr_european(
    spot: float,
    strike: float,
    tau: float,
    vol: float,
    rate: float,
    div: float,
    opt_type: int,
    steps: int,
) -> float:
    """Cox-Ross-Rubinstein European option pricing."""
    dt = tau / steps
    u = np.exp(vol * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((rate - div) * dt) - d) / (u - d)
    disc = np.exp(-rate * dt)
    
    # Terminal payoffs
    prices = np.empty(steps + 1, dtype=np.float64)
    for i in range(steps + 1):
        S_T = spot * (u ** (steps - i)) * (d ** i)
        if opt_type == 1:
            prices[i] = max(S_T - strike, 0.0)
        else:
            prices[i] = max(strike - S_T, 0.0)
    
    # Backward induction
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            prices[i] = disc * (p * prices[i] + (1 - p) * prices[i + 1])
    
    return prices[0]


@jit(nopython=True, cache=True, fastmath=True)
def _crr_american(
    spot: float,
    strike: float,
    tau: float,
    vol: float,
    rate: float,
    div: float,
    opt_type: int,
    steps: int,
) -> Tuple[float, np.ndarray]:
    """CRR American option with early exercise boundary."""
    dt = tau / steps
    u = np.exp(vol * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((rate - div) * dt) - d) / (u - d)
    disc = np.exp(-rate * dt)
    
    # Early exercise boundary
    boundary = np.zeros(steps + 1, dtype=np.float64)
    
    # Terminal payoffs
    prices = np.empty(steps + 1, dtype=np.float64)
    for i in range(steps + 1):
        S_T = spot * (u ** (steps - i)) * (d ** i)
        if opt_type == 1:
            prices[i] = max(S_T - strike, 0.0)
        else:
            prices[i] = max(strike - S_T, 0.0)
    
    # Backward induction with early exercise
    for step in range(steps - 1, -1, -1):
        boundary_found = False
        for i in range(step + 1):
            S_t = spot * (u ** (step - i)) * (d ** i)
            continuation = disc * (p * prices[i] + (1 - p) * prices[i + 1])
            
            if opt_type == 1:
                exercise = max(S_t - strike, 0.0)
            else:
                exercise = max(strike - S_t, 0.0)
            
            prices[i] = max(continuation, exercise)
            
            # Track exercise boundary
            if not boundary_found and exercise > continuation:
                boundary[step] = S_t
                boundary_found = True
    
    return prices[0], boundary


@jit(nopython=True, cache=True, fastmath=True)
def _crr_greeks(
    spot: float,
    strike: float,
    tau: float,
    vol: float,
    rate: float,
    div: float,
    opt_type: int,
    steps: int,
    american: bool,
) -> Tuple[float, float, float]:
    """Compute delta and gamma from tree."""
    dt = tau / steps
    u = np.exp(vol * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((rate - div) * dt) - d) / (u - d)
    disc = np.exp(-rate * dt)
    
    # Build tree to step 2 for Greeks
    prices = np.empty((3,), dtype=np.float64)
    
    # Full tree backward
    tree = np.empty(steps + 1, dtype=np.float64)
    for i in range(steps + 1):
        S_T = spot * (u ** (steps - i)) * (d ** i)
        if opt_type == 1:
            tree[i] = max(S_T - strike, 0.0)
        else:
            tree[i] = max(strike - S_T, 0.0)
    
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            continuation = disc * (p * tree[i] + (1 - p) * tree[i + 1])
            
            if american:
                S_t = spot * (u ** (step - i)) * (d ** i)
                if opt_type == 1:
                    exercise = max(S_t - strike, 0.0)
                else:
                    exercise = max(strike - S_t, 0.0)
                tree[i] = max(continuation, exercise)
            else:
                tree[i] = continuation
        
        if step == 2:
            prices[0] = tree[0]  # S*u^2
            prices[1] = tree[1]  # S*u*d = S
            prices[2] = tree[2]  # S*d^2
        elif step == 1:
            f_u = tree[0]
            f_d = tree[1]
        elif step == 0:
            price = tree[0]
    
    # Greeks
    S_u = spot * u
    S_d = spot * d
    delta = (f_u - f_d) / (S_u - S_d)
    
    S_uu = spot * u * u
    S_dd = spot * d * d
    delta_u = (prices[0] - prices[1]) / (S_uu - spot)
    delta_d = (prices[1] - prices[2]) / (spot - S_dd)
    gamma = (delta_u - delta_d) / (0.5 * (S_uu - S_dd))
    
    return price, delta, gamma


class BinomialTreePricer:
    """
    Binomial tree option pricer.
    
    Features:
    - European and American options
    - Early exercise boundary tracking
    - Greeks computation from tree
    - Richardson extrapolation
    """
    
    def __init__(self, steps: int = 500, dividend_yield: float = 0.0):
        self.steps = steps
        self.div = dividend_yield
        # Warm up
        _ = _crr_european(100.0, 100.0, 0.25, 0.2, 0.05, 0.0, 1, 10)
    
    def price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        option_type: OptionType = OptionType.CALL,
        american: bool = True,
        steps: int = None,
    ) -> float:
        """Price an option using binomial tree."""
        n = steps or self.steps
        
        if american:
            price, _ = _crr_american(
                spot, strike, time_to_expiry, volatility, rate,
                self.div, int(option_type), n
            )
        else:
            price = _crr_european(
                spot, strike, time_to_expiry, volatility, rate,
                self.div, int(option_type), n
            )
        
        return price
    
    def price_with_extrapolation(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        option_type: OptionType = OptionType.CALL,
        american: bool = True,
    ) -> float:
        """Price with Richardson extrapolation for improved accuracy."""
        n = self.steps
        
        # Price at n and n/2 steps
        p1 = self.price(spot, strike, time_to_expiry, volatility, rate,
                       option_type, american, n)
        p2 = self.price(spot, strike, time_to_expiry, volatility, rate,
                       option_type, american, n // 2)
        
        # Richardson extrapolation
        return 2 * p1 - p2
    
    def price_american_with_boundary(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        option_type: OptionType = OptionType.PUT,
    ) -> TreeResult:
        """Price American option and return early exercise boundary."""
        price, boundary = _crr_american(
            spot, strike, time_to_expiry, volatility, rate,
            self.div, int(option_type), self.steps
        )
        
        _, delta, gamma = _crr_greeks(
            spot, strike, time_to_expiry, volatility, rate,
            self.div, int(option_type), self.steps, True
        )
        
        return TreeResult(
            price=price,
            delta=delta,
            gamma=gamma,
            early_exercise_boundary=boundary,
        )
    
    def early_exercise_premium(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        rate: float,
        option_type: OptionType = OptionType.PUT,
    ) -> float:
        """Compute early exercise premium (American - European)."""
        american = self.price(spot, strike, time_to_expiry, volatility, rate,
                            option_type, american=True)
        european = self.price(spot, strike, time_to_expiry, volatility, rate,
                            option_type, american=False)
        return american - european
