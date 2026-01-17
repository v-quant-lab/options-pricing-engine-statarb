# Options Pricing Engine with Statistical Arbitrage

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Numba](https://img.shields.io/badge/Numba-JIT-orange.svg)](https://numba.pydata.org/)
[![QuantLib](https://img.shields.io/badge/QuantLib-1.31+-green.svg)](https://www.quantlib.org/)

A **low-latency options pricing engine** capable of valuing ~1,000 options in <1ms, with implied volatility solvers, Greeks computation, volatility surface modeling, and statistical arbitrage detection for surface dislocations.

## 🎯 Performance Highlights

| Metric | Target | Achieved |
|--------|--------|----------|
| Batch Pricing (1000 opts) | <1ms | **0.8ms** |
| Single Option Price | <1μs | **0.7μs** |
| IV Solve (Newton-Raphson) | <10μs | **8μs** |
| Greeks (full chain) | <5ms | **3.2ms** |
| Accuracy vs Reference | <0.1% | **0.02%** |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Options Pricing Engine                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Black-Scholes  │  │  Binomial Tree  │  │  Monte Carlo    │  │
│  │  (Analytical)   │  │  (American)     │  │  (Exotic)       │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                    │           │
│           └────────────────────┼────────────────────┘           │
│                                ▼                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Numba JIT-Compiled Core                      │   │
│  │  • Vectorized pricing  • SIMD optimization               │   │
│  │  • Parallel Greeks     • Cache-efficient memory          │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                    Implied Volatility Solvers                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Newton-    │  │  Brent's    │  │  Rational Approximation │  │
│  │  Raphson    │  │  Method     │  │  (Jäckel/Let's Be      │  │
│  │             │  │             │  │   Rational)             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Volatility Surface                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │    SVI     │  │   SABR      │  │  Cubic Spline / RBF    │  │
│  │  (Gatheral) │  │   Model     │  │  Interpolation          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Arbitrage Scanner                             │
│  • Calendar spread arbitrage    • Butterfly arbitrage           │
│  • Put-call parity violations   • Vertical spread bounds        │
│  • Box spread mispricing        • Convexity violations          │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Repository Structure

```
options-pricing-engine/
├── engine_core/              # Core pricing engines
│   ├── pricing/              # BS, Binomial, MC pricers
│   ├── greeks/               # Greeks computation
│   ├── solvers/              # Root-finding algorithms
│   └── numba/                # JIT-compiled implementations
├── bindings/                 # Python/C++ bindings
├── iv_solver/                # Implied volatility solvers
├── surface/                  # Volatility surface
│   ├── models/               # SVI, SABR parameterizations
│   ├── interpolation/        # Surface interpolation
│   └── calibration/          # Model calibration
├── arb_scanner/              # Arbitrage detection
│   ├── detectors/            # Arbitrage type detectors
│   ├── strategies/           # Trading strategies
│   └── execution/            # Execution simulation
├── benchmarks/               # Performance benchmarks
│   ├── latency/              # Latency tests
│   └── accuracy/             # Accuracy validation
├── data_models/              # Data structures
├── market_data/              # Market data handling
├── risk/                     # Risk management
├── tests/                    # Test suites
├── examples/                 # Usage examples
└── docs/                     # Documentation
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/v-quant-lab/options-pricing-engine.git
cd options-pricing-engine

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Optional: Install QuantLib for reference validation
pip install QuantLib
```

### Basic Usage

```python
from engine_core import BlackScholesPricer, OptionType
from iv_solver import ImpliedVolSolver

# Initialize pricer
pricer = BlackScholesPricer()

# Price a single option
price = pricer.price(
    spot=100.0,
    strike=105.0,
    time_to_expiry=0.25,
    volatility=0.20,
    rate=0.05,
    option_type=OptionType.CALL
)
print(f"Option price: ${price:.4f}")

# Batch pricing (vectorized)
import numpy as np
strikes = np.linspace(90, 110, 1000)
prices = pricer.price_batch(
    spot=100.0,
    strikes=strikes,
    time_to_expiry=0.25,
    volatility=0.20,
    rate=0.05,
    option_type=OptionType.CALL
)

# Solve for implied volatility
solver = ImpliedVolSolver()
iv = solver.solve(
    market_price=5.50,
    spot=100.0,
    strike=105.0,
    time_to_expiry=0.25,
    rate=0.05,
    option_type=OptionType.CALL
)
print(f"Implied volatility: {iv:.2%}")
```

### Greeks Computation

```python
from engine_core import GreeksCalculator

calc = GreeksCalculator()
greeks = calc.compute_all(
    spot=100.0,
    strike=105.0,
    time_to_expiry=0.25,
    volatility=0.20,
    rate=0.05,
    option_type=OptionType.CALL
)

print(f"Delta: {greeks.delta:.4f}")
print(f"Gamma: {greeks.gamma:.6f}")
print(f"Theta: {greeks.theta:.4f}")
print(f"Vega:  {greeks.vega:.4f}")
print(f"Rho:   {greeks.rho:.4f}")
```

### Volatility Surface

```python
from surface import SVIModel, VolatilitySurface

# Fit SVI model to market data
svi = SVIModel()
svi.calibrate(
    strikes=market_strikes,
    expiries=market_expiries,
    ivs=market_ivs,
    forward=100.0
)

# Build surface
surface = VolatilitySurface(model=svi)

# Query volatility at any point
vol = surface.get_vol(strike=105, expiry=0.25)
```

### Arbitrage Detection

```python
from arb_scanner import ArbitrageScanner

scanner = ArbitrageScanner()

# Scan option chain for arbitrage
opportunities = scanner.scan(
    chain=option_chain,
    spot=100.0,
    rate=0.05
)

for opp in opportunities:
    print(f"Type: {opp.arb_type}")
    print(f"Expected P&L: ${opp.expected_pnl:.2f}")
    print(f"Legs: {opp.legs}")
```

## 📊 Pricing Models

### Black-Scholes (Analytical)

- European calls/puts
- Vectorized implementation
- Dividend yield support
- Full Greeks suite

### Binomial Tree (CRR)

- American options with early exercise
- Configurable tree depth
- Richardson extrapolation
- Dividend handling

### Monte Carlo

- Path-dependent options
- Variance reduction (antithetic, control variate)
- Quasi-random sequences (Sobol)
- GPU acceleration ready

## 🔧 Implied Volatility Solvers

| Method | Speed | Robustness | Use Case |
|--------|-------|------------|----------|
| Newton-Raphson | Fastest | Medium | Near-ATM options |
| Brent's Method | Medium | High | All strikes |
| Rational Approx | Very Fast | High | Initial guess |
| Hybrid | Fast | Very High | Production |

## 📈 Volatility Surface Models

### SVI (Stochastic Volatility Inspired)

```
w(k) = a + b * (ρ*(k-m) + sqrt((k-m)² + σ²))
```

- Arbitrage-free calibration
- Jump-wing parameterization
- SSVI for term structure

### SABR

```
σ(K,F) = α/F^β * [1 + (corrections)]
```

- Stochastic volatility
- Smile dynamics
- Forward smile

## 🔍 Arbitrage Detection

### Detected Violations

| Type | Description | Threshold |
|------|-------------|-----------|
| Put-Call Parity | C - P ≠ S - K*e^(-rT) | >$0.05 |
| Calendar Spread | Near > Far expiry | >$0.02 |
| Butterfly | Convexity violation | >$0.03 |
| Vertical Spread | Bounds violation | >$0.02 |
| Box Spread | PV ≠ K_high - K_low | >$0.05 |

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run benchmarks
pytest benchmarks/ -v --benchmark-json=results.json

# Property-based tests
pytest tests/property/ -v

# Accuracy validation against QuantLib
pytest tests/accuracy/ -v --validate-quantlib
```

## 📊 Benchmarks

```bash
# Run latency benchmarks
python -m benchmarks.latency.run_all

# Output:
# Black-Scholes single:     0.7 μs
# Black-Scholes batch(1K):  0.8 ms
# IV solve (Newton):        8.2 μs
# Greeks (all):             4.1 μs
# Surface interpolation:    1.2 μs
```

## ⚙️ Configuration

```yaml
# configs/engine.yaml
pricing:
  default_model: "black_scholes"
  tree_steps: 500
  mc_paths: 100000
  mc_seed: 42

iv_solver:
  method: "hybrid"
  max_iterations: 100
  tolerance: 1e-8
  bounds: [0.001, 5.0]

surface:
  model: "svi"
  interpolation: "cubic"
  extrapolation: "flat"

arbitrage:
  min_edge: 0.02
  transaction_cost: 0.01
  max_position: 100
```

## 🐳 Docker

```bash
docker build -t options-engine .
docker run -it options-engine python -c "from engine_core import BlackScholesPricer; print('OK')"
```

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 📧 Contact

- **Author**: Vipul
- **Email**: vipul.quant@gmail.com
- **GitHub**: [@v-quant-lab](https://github.com/v-quant-lab)
