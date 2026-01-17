"""Unit tests for pricing engine."""

import pytest
import numpy as np
from engine_core.pricing.black_scholes import BlackScholesPricer, OptionType, GreeksCalculator
from engine_core.pricing.binomial import BinomialTreePricer
from iv_solver.solver import ImpliedVolSolver


class TestBlackScholesPricer:
    
    @pytest.fixture
    def pricer(self):
        return BlackScholesPricer()
    
    def test_call_price_atm(self, pricer):
        price = pricer.price(100, 100, 1.0, 0.2, 0.05, OptionType.CALL)
        assert 10 < price < 15
    
    def test_put_price_atm(self, pricer):
        price = pricer.price(100, 100, 1.0, 0.2, 0.05, OptionType.PUT)
        assert 5 < price < 10
    
    def test_put_call_parity(self, pricer):
        S, K, T, vol, r = 100, 100, 1.0, 0.2, 0.05
        call = pricer.price(S, K, T, vol, r, OptionType.CALL)
        put = pricer.price(S, K, T, vol, r, OptionType.PUT)
        
        parity = call - put - (S - K * np.exp(-r * T))
        assert abs(parity) < 1e-10
    
    def test_batch_pricing(self, pricer):
        strikes = np.linspace(80, 120, 1000)
        prices = pricer.price_batch(100, strikes, 0.25, 0.2, 0.05, OptionType.CALL)
        
        assert len(prices) == 1000
        assert all(p >= 0 for p in prices)
        assert prices[0] > prices[-1]  # Lower strike = higher call price
    
    def test_greeks_delta_bounds(self, pricer):
        result = pricer.greeks(100, 100, 0.5, 0.25, 0.05, OptionType.CALL)
        assert 0 < result.delta < 1
        
        result = pricer.greeks(100, 100, 0.5, 0.25, 0.05, OptionType.PUT)
        assert -1 < result.delta < 0
    
    def test_expiry_intrinsic(self, pricer):
        call_itm = pricer.price(110, 100, 0.0, 0.2, 0.05, OptionType.CALL)
        assert abs(call_itm - 10) < 0.01
        
        put_itm = pricer.price(90, 100, 0.0, 0.2, 0.05, OptionType.PUT)
        assert abs(put_itm - 10) < 0.01


class TestBinomialTreePricer:
    
    @pytest.fixture
    def pricer(self):
        return BinomialTreePricer(steps=500)
    
    def test_european_matches_bs(self, pricer):
        bs = BlackScholesPricer()
        
        bs_price = bs.price(100, 100, 0.5, 0.2, 0.05, OptionType.CALL)
        tree_price = pricer.price(100, 100, 0.5, 0.2, 0.05, OptionType.CALL, american=False)
        
        assert abs(bs_price - tree_price) < 0.05
    
    def test_american_put_premium(self, pricer):
        american = pricer.price(100, 100, 1.0, 0.3, 0.05, OptionType.PUT, american=True)
        european = pricer.price(100, 100, 1.0, 0.3, 0.05, OptionType.PUT, american=False)
        
        assert american >= european


class TestImpliedVolSolver:
    
    @pytest.fixture
    def solver(self):
        return ImpliedVolSolver()
    
    def test_iv_roundtrip(self, solver):
        pricer = BlackScholesPricer()
        true_vol = 0.25
        
        price = pricer.price(100, 100, 0.5, true_vol, 0.05, OptionType.CALL)
        result = solver.solve(price, 100, 100, 0.5, 0.05, OptionType.CALL)
        
        assert result.converged
        assert abs(result.iv - true_vol) < 1e-6
    
    def test_iv_otm_call(self, solver):
        result = solver.solve(1.0, 100, 120, 0.25, 0.05, OptionType.CALL)
        assert result.converged
        assert 0.1 < result.iv < 1.0
    
    def test_batch_iv(self, solver):
        pricer = BlackScholesPricer()
        strikes = np.array([95, 100, 105])
        prices = pricer.price_batch(100, strikes, 0.25, 0.2, 0.05, OptionType.CALL)
        
        ivs = solver.solve_batch(prices, 100, strikes, 0.25, 0.05, np.ones(3))
        
        assert len(ivs) == 3
        assert all(abs(iv - 0.2) < 0.001 for iv in ivs)
