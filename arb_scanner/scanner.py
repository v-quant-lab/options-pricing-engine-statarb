"""
Arbitrage Scanner

Detects statistical arbitrage opportunities in options markets.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple
from enum import Enum
from loguru import logger

from engine_core.pricing.black_scholes import BlackScholesPricer, OptionType


class ArbitrageType(Enum):
    PUT_CALL_PARITY = "put_call_parity"
    CALENDAR_SPREAD = "calendar_spread"
    BUTTERFLY = "butterfly"
    VERTICAL_SPREAD = "vertical_spread"
    BOX_SPREAD = "box_spread"
    CONVEXITY = "convexity"


@dataclass
class OptionQuote:
    strike: float
    expiry: float
    option_type: OptionType
    bid: float
    ask: float
    iv: Optional[float] = None
    
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class ArbitrageOpportunity:
    arb_type: ArbitrageType
    expected_pnl: float
    max_pnl: float
    legs: List[Tuple[str, OptionQuote, int]]  # (action, quote, quantity)
    confidence: float
    edge: float
    transaction_costs: float
    net_pnl: float
    
    def __repr__(self) -> str:
        return f"Arb({self.arb_type.value}, edge=${self.edge:.3f}, net=${self.net_pnl:.3f})"


@dataclass
class OptionChain:
    spot: float
    expiries: List[float]
    strikes: List[float]
    calls: List[OptionQuote]
    puts: List[OptionQuote]
    rate: float = 0.05
    
    def get_call(self, strike: float, expiry: float) -> Optional[OptionQuote]:
        for c in self.calls:
            if abs(c.strike - strike) < 0.01 and abs(c.expiry - expiry) < 0.001:
                return c
        return None
    
    def get_put(self, strike: float, expiry: float) -> Optional[OptionQuote]:
        for p in self.puts:
            if abs(p.strike - strike) < 0.01 and abs(p.expiry - expiry) < 0.001:
                return p
        return None


class ArbitrageScanner:
    """
    Scans option chains for arbitrage opportunities.
    
    Detects:
    - Put-call parity violations
    - Calendar spread arbitrage
    - Butterfly arbitrage
    - Vertical spread bound violations
    - Box spread mispricing
    - Convexity violations
    """
    
    def __init__(
        self,
        min_edge: float = 0.02,
        transaction_cost: float = 0.01,
        max_position: int = 100,
    ):
        self.min_edge = min_edge
        self.txn_cost = transaction_cost
        self.max_position = max_position
        self.pricer = BlackScholesPricer()
    
    def scan(self, chain: OptionChain) -> List[ArbitrageOpportunity]:
        """Scan chain for all arbitrage types."""
        opportunities = []
        
        opportunities.extend(self._scan_put_call_parity(chain))
        opportunities.extend(self._scan_calendar_spread(chain))
        opportunities.extend(self._scan_butterfly(chain))
        opportunities.extend(self._scan_vertical_spread(chain))
        opportunities.extend(self._scan_box_spread(chain))
        
        # Filter by minimum edge after costs
        opportunities = [o for o in opportunities if o.net_pnl > self.min_edge]
        
        # Sort by net P&L
        opportunities.sort(key=lambda x: x.net_pnl, reverse=True)
        
        logger.info(f"Found {len(opportunities)} arbitrage opportunities")
        return opportunities
    
    def _scan_put_call_parity(self, chain: OptionChain) -> List[ArbitrageOpportunity]:
        """Detect put-call parity violations."""
        opportunities = []
        
        for expiry in chain.expiries:
            df = np.exp(-chain.rate * expiry)
            
            for strike in chain.strikes:
                call = chain.get_call(strike, expiry)
                put = chain.get_put(strike, expiry)
                
                if call is None or put is None:
                    continue
                
                # Theoretical: C - P = S - K*exp(-rT)
                theoretical_diff = chain.spot - strike * df
                
                # Market difference (using bids/asks for direction)
                # Long synthetic: buy call, sell put
                long_synthetic_cost = call.ask - put.bid
                # Short synthetic: sell call, buy put
                short_synthetic_cost = put.ask - call.bid
                
                # Check for long synthetic arbitrage: C_ask - P_bid < S - K*df
                edge_long = theoretical_diff - long_synthetic_cost
                if edge_long > self.min_edge:
                    txn = 4 * self.txn_cost  # 2 options + stock
                    opportunities.append(ArbitrageOpportunity(
                        arb_type=ArbitrageType.PUT_CALL_PARITY,
                        expected_pnl=edge_long,
                        max_pnl=edge_long,
                        legs=[
                            ("BUY", call, 1),
                            ("SELL", put, 1),
                            ("SELL_STOCK", None, 100),
                        ],
                        confidence=0.95,
                        edge=edge_long,
                        transaction_costs=txn,
                        net_pnl=edge_long - txn,
                    ))
                
                # Check for short synthetic arbitrage
                edge_short = short_synthetic_cost - theoretical_diff
                if edge_short > self.min_edge:
                    txn = 4 * self.txn_cost
                    opportunities.append(ArbitrageOpportunity(
                        arb_type=ArbitrageType.PUT_CALL_PARITY,
                        expected_pnl=edge_short,
                        max_pnl=edge_short,
                        legs=[
                            ("SELL", call, 1),
                            ("BUY", put, 1),
                            ("BUY_STOCK", None, 100),
                        ],
                        confidence=0.95,
                        edge=edge_short,
                        transaction_costs=txn,
                        net_pnl=edge_short - txn,
                    ))
        
        return opportunities
    
    def _scan_calendar_spread(self, chain: OptionChain) -> List[ArbitrageOpportunity]:
        """Detect calendar spread arbitrage."""
        opportunities = []
        
        sorted_expiries = sorted(chain.expiries)
        
        for i in range(len(sorted_expiries) - 1):
            near_exp = sorted_expiries[i]
            far_exp = sorted_expiries[i + 1]
            
            for strike in chain.strikes:
                # Check calls
                near_call = chain.get_call(strike, near_exp)
                far_call = chain.get_call(strike, far_exp)
                
                if near_call and far_call:
                    # Far expiry should be worth more
                    # Arb if near_bid > far_ask
                    edge = near_call.bid - far_call.ask
                    if edge > self.min_edge:
                        txn = 2 * self.txn_cost
                        opportunities.append(ArbitrageOpportunity(
                            arb_type=ArbitrageType.CALENDAR_SPREAD,
                            expected_pnl=edge,
                            max_pnl=edge,
                            legs=[
                                ("SELL", near_call, 1),
                                ("BUY", far_call, 1),
                            ],
                            confidence=0.90,
                            edge=edge,
                            transaction_costs=txn,
                            net_pnl=edge - txn,
                        ))
                
                # Check puts
                near_put = chain.get_put(strike, near_exp)
                far_put = chain.get_put(strike, far_exp)
                
                if near_put and far_put:
                    edge = near_put.bid - far_put.ask
                    if edge > self.min_edge:
                        txn = 2 * self.txn_cost
                        opportunities.append(ArbitrageOpportunity(
                            arb_type=ArbitrageType.CALENDAR_SPREAD,
                            expected_pnl=edge,
                            max_pnl=edge,
                            legs=[
                                ("SELL", near_put, 1),
                                ("BUY", far_put, 1),
                            ],
                            confidence=0.90,
                            edge=edge,
                            transaction_costs=txn,
                            net_pnl=edge - txn,
                        ))
        
        return opportunities
    
    def _scan_butterfly(self, chain: OptionChain) -> List[ArbitrageOpportunity]:
        """Detect butterfly arbitrage (convexity violations)."""
        opportunities = []
        
        sorted_strikes = sorted(chain.strikes)
        
        for expiry in chain.expiries:
            for i in range(len(sorted_strikes) - 2):
                K1, K2, K3 = sorted_strikes[i], sorted_strikes[i+1], sorted_strikes[i+2]
                
                # Check if K2 is midpoint (or close)
                if abs(K2 - (K1 + K3) / 2) > 0.01:
                    continue
                
                # Call butterfly
                c1 = chain.get_call(K1, expiry)
                c2 = chain.get_call(K2, expiry)
                c3 = chain.get_call(K3, expiry)
                
                if c1 and c2 and c3:
                    # Long butterfly: buy K1, sell 2*K2, buy K3
                    # Must have non-negative value
                    cost = c1.ask - 2 * c2.bid + c3.ask
                    
                    if cost < -self.min_edge:
                        edge = -cost
                        txn = 4 * self.txn_cost
                        opportunities.append(ArbitrageOpportunity(
                            arb_type=ArbitrageType.BUTTERFLY,
                            expected_pnl=edge,
                            max_pnl=K2 - K1 + edge,
                            legs=[
                                ("BUY", c1, 1),
                                ("SELL", c2, 2),
                                ("BUY", c3, 1),
                            ],
                            confidence=0.85,
                            edge=edge,
                            transaction_costs=txn,
                            net_pnl=edge - txn,
                        ))
        
        return opportunities
    
    def _scan_vertical_spread(self, chain: OptionChain) -> List[ArbitrageOpportunity]:
        """Detect vertical spread bound violations."""
        opportunities = []
        
        sorted_strikes = sorted(chain.strikes)
        
        for expiry in chain.expiries:
            df = np.exp(-chain.rate * expiry)
            
            for i in range(len(sorted_strikes) - 1):
                K1, K2 = sorted_strikes[i], sorted_strikes[i+1]
                
                c1 = chain.get_call(K1, expiry)
                c2 = chain.get_call(K2, expiry)
                
                if c1 and c2:
                    # Call spread must be <= K2 - K1 (discounted)
                    max_value = (K2 - K1) * df
                    spread_cost = c1.ask - c2.bid  # Long K1, short K2
                    
                    if spread_cost > max_value + self.min_edge:
                        edge = spread_cost - max_value
                        txn = 2 * self.txn_cost
                        opportunities.append(ArbitrageOpportunity(
                            arb_type=ArbitrageType.VERTICAL_SPREAD,
                            expected_pnl=edge,
                            max_pnl=edge,
                            legs=[
                                ("SELL", c1, 1),
                                ("BUY", c2, 1),
                            ],
                            confidence=0.80,
                            edge=edge,
                            transaction_costs=txn,
                            net_pnl=edge - txn,
                        ))
        
        return opportunities
    
    def _scan_box_spread(self, chain: OptionChain) -> List[ArbitrageOpportunity]:
        """Detect box spread mispricing."""
        opportunities = []
        
        sorted_strikes = sorted(chain.strikes)
        
        for expiry in chain.expiries:
            df = np.exp(-chain.rate * expiry)
            
            for i in range(len(sorted_strikes) - 1):
                K1, K2 = sorted_strikes[i], sorted_strikes[i+1]
                
                c1 = chain.get_call(K1, expiry)
                c2 = chain.get_call(K2, expiry)
                p1 = chain.get_put(K1, expiry)
                p2 = chain.get_put(K2, expiry)
                
                if not all([c1, c2, p1, p2]):
                    continue
                
                # Box spread = (K2 - K1) * df
                theoretical = (K2 - K1) * df
                
                # Long box: buy c1, sell c2, sell p1, buy p2
                long_box_cost = c1.ask - c2.bid - p1.bid + p2.ask
                
                edge = theoretical - long_box_cost
                if edge > self.min_edge:
                    txn = 4 * self.txn_cost
                    opportunities.append(ArbitrageOpportunity(
                        arb_type=ArbitrageType.BOX_SPREAD,
                        expected_pnl=edge,
                        max_pnl=edge,
                        legs=[
                            ("BUY", c1, 1),
                            ("SELL", c2, 1),
                            ("SELL", p1, 1),
                            ("BUY", p2, 1),
                        ],
                        confidence=0.95,
                        edge=edge,
                        transaction_costs=txn,
                        net_pnl=edge - txn,
                    ))
        
        return opportunities
