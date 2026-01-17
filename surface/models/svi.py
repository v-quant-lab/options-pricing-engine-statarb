"""
Volatility Surface Models

SVI (Stochastic Volatility Inspired) and interpolation methods.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from loguru import logger


@dataclass
class SVIParams:
    """SVI raw parameterization."""
    a: float      # Level
    b: float      # Angle of wings
    rho: float    # Rotation (-1, 1)
    m: float      # Translation
    sigma: float  # ATM curvature
    
    def total_variance(self, k: float) -> float:
        """Compute total variance w(k)."""
        return self.a + self.b * (self.rho * (k - self.m) + 
               np.sqrt((k - self.m)**2 + self.sigma**2))
    
    def implied_vol(self, k: float, tau: float) -> float:
        """Compute implied volatility."""
        w = self.total_variance(k)
        return np.sqrt(max(w / tau, 1e-8))


@dataclass  
class SSVIParams:
    """SSVI (Surface SVI) parameterization."""
    rho: float
    phi_params: Tuple[float, float]  # (theta, eta) for ATM total variance
    
    def atm_variance(self, tau: float) -> float:
        """ATM total variance as function of time."""
        theta, eta = self.phi_params
        return theta * tau ** eta
    
    def total_variance(self, k: float, tau: float) -> float:
        """SSVI total variance."""
        theta_t = self.atm_variance(tau)
        phi = theta_t / 2 * (1 + self.rho * k / np.sqrt(k**2 + 1))
        return phi


class SVICalibrator:
    """
    SVI model calibration with arbitrage-free constraints.
    """
    
    BOUNDS = {
        'a': (-0.5, 0.5),
        'b': (0.001, 1.0),
        'rho': (-0.999, 0.999),
        'm': (-0.5, 0.5),
        'sigma': (0.001, 1.0),
    }
    
    def __init__(self, enforce_arbitrage_free: bool = True):
        self.enforce_arbitrage_free = enforce_arbitrage_free
    
    def _objective(
        self,
        params: np.ndarray,
        log_strikes: np.ndarray,
        market_vars: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Weighted sum of squared errors."""
        a, b, rho, m, sigma = params
        
        model_vars = a + b * (rho * (log_strikes - m) + 
                             np.sqrt((log_strikes - m)**2 + sigma**2))
        
        sse = np.sum(weights * (model_vars - market_vars)**2)
        
        # Arbitrage penalty
        if self.enforce_arbitrage_free:
            # Butterfly arbitrage: d²w/dk² >= 0
            min_var = a + b * sigma * np.sqrt(1 - rho**2)
            if min_var < 0:
                sse += 1000 * min_var**2
            
            # Calendar arbitrage: w must be increasing in t (handled at surface level)
        
        return sse
    
    def calibrate(
        self,
        log_strikes: np.ndarray,
        market_ivs: np.ndarray,
        tau: float,
        weights: Optional[np.ndarray] = None,
        method: str = "differential_evolution",
    ) -> SVIParams:
        """Calibrate SVI to market data."""
        market_vars = market_ivs**2 * tau
        
        if weights is None:
            weights = np.ones_like(log_strikes)
        
        bounds = [
            self.BOUNDS['a'],
            self.BOUNDS['b'],
            self.BOUNDS['rho'],
            self.BOUNDS['m'],
            self.BOUNDS['sigma'],
        ]
        
        if method == "differential_evolution":
            result = differential_evolution(
                self._objective,
                bounds,
                args=(log_strikes, market_vars, weights),
                maxiter=500,
                tol=1e-8,
                seed=42,
            )
        else:
            x0 = [0.04, 0.1, -0.3, 0.0, 0.1]
            result = minimize(
                self._objective,
                x0,
                args=(log_strikes, market_vars, weights),
                method='L-BFGS-B',
                bounds=bounds,
            )
        
        return SVIParams(
            a=result.x[0],
            b=result.x[1],
            rho=result.x[2],
            m=result.x[3],
            sigma=result.x[4],
        )


class VolatilitySurface:
    """
    Volatility surface with interpolation and extrapolation.
    
    Features:
    - SVI parameterization per expiry
    - Smooth interpolation across strikes and expiries
    - Arbitrage-free extrapolation
    - Greeks from surface
    """
    
    def __init__(self):
        self.slices: Dict[float, SVIParams] = {}
        self.expiries: List[float] = []
        self._calibrator = SVICalibrator()
    
    def add_slice(
        self,
        tau: float,
        strikes: np.ndarray,
        ivs: np.ndarray,
        forward: float,
    ) -> None:
        """Add calibrated slice to surface."""
        log_strikes = np.log(strikes / forward)
        params = self._calibrator.calibrate(log_strikes, ivs, tau)
        
        self.slices[tau] = params
        self.expiries = sorted(self.slices.keys())
        
        logger.debug(f"Added slice tau={tau}: a={params.a:.4f}, b={params.b:.4f}")
    
    def get_vol(
        self,
        strike: float,
        tau: float,
        forward: float,
    ) -> float:
        """Get interpolated volatility."""
        k = np.log(strike / forward)
        
        if tau in self.slices:
            return self.slices[tau].implied_vol(k, tau)
        
        # Linear interpolation in variance
        tau_below = max([t for t in self.expiries if t <= tau], default=None)
        tau_above = min([t for t in self.expiries if t >= tau], default=None)
        
        if tau_below is None:
            return self.slices[tau_above].implied_vol(k, tau)
        if tau_above is None:
            return self.slices[tau_below].implied_vol(k, tau)
        
        var_below = self.slices[tau_below].total_variance(k)
        var_above = self.slices[tau_above].total_variance(k)
        
        weight = (tau - tau_below) / (tau_above - tau_below)
        var_interp = var_below + weight * (var_above - var_below)
        
        return np.sqrt(max(var_interp / tau, 1e-8))
    
    def get_vol_grid(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,
        forward: float,
    ) -> np.ndarray:
        """Get volatility grid."""
        grid = np.empty((len(expiries), len(strikes)))
        
        for i, tau in enumerate(expiries):
            for j, K in enumerate(strikes):
                grid[i, j] = self.get_vol(K, tau, forward)
        
        return grid
    
    def local_vol(
        self,
        strike: float,
        tau: float,
        forward: float,
        dK: float = 0.01,
        dT: float = 0.01,
    ) -> float:
        """Compute Dupire local volatility."""
        K = strike
        T = tau
        
        # Numerical derivatives
        sigma = self.get_vol(K, T, forward)
        
        sigma_K_up = self.get_vol(K * (1 + dK), T, forward)
        sigma_K_dn = self.get_vol(K * (1 - dK), T, forward)
        d_sigma_dK = (sigma_K_up - sigma_K_dn) / (2 * K * dK)
        d2_sigma_dK2 = (sigma_K_up - 2 * sigma + sigma_K_dn) / (K * dK)**2
        
        if T + dT <= max(self.expiries):
            sigma_T_up = self.get_vol(K, T + dT, forward)
            d_sigma_dT = (sigma_T_up - sigma) / dT
        else:
            d_sigma_dT = 0.0
        
        # Dupire formula
        w = sigma**2 * T
        dw_dT = 2 * sigma * T * d_sigma_dT + sigma**2
        dw_dK = 2 * sigma * T * d_sigma_dK
        d2w_dK2 = 2 * T * (d_sigma_dK**2 + sigma * d2_sigma_dK2)
        
        k = np.log(K / forward)
        
        numerator = dw_dT
        denominator = (1 - k * dw_dK / w + 0.25 * (-0.25 - 1/w + k**2/w**2) * dw_dK**2 +
                      0.5 * d2w_dK2)
        
        if denominator <= 0:
            return sigma
        
        local_var = numerator / denominator
        return np.sqrt(max(local_var, 1e-8))
    
    def is_arbitrage_free(self, tolerance: float = 1e-6) -> Tuple[bool, List[str]]:
        """Check for arbitrage violations."""
        violations = []
        
        # Check each slice for butterfly arbitrage
        for tau, params in self.slices.items():
            min_var = params.a + params.b * params.sigma * np.sqrt(1 - params.rho**2)
            if min_var < -tolerance:
                violations.append(f"Butterfly violation at tau={tau}: min_var={min_var:.6f}")
        
        # Check calendar arbitrage
        for i in range(len(self.expiries) - 1):
            t1, t2 = self.expiries[i], self.expiries[i+1]
            
            for k in np.linspace(-0.5, 0.5, 20):
                w1 = self.slices[t1].total_variance(k)
                w2 = self.slices[t2].total_variance(k)
                
                if w2 < w1 - tolerance:
                    violations.append(f"Calendar violation at k={k:.2f}: w({t1})={w1:.4f} > w({t2})={w2:.4f}")
        
        return len(violations) == 0, violations
