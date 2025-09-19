# Trading Strategy Framework with Advanced Performance Metrics

A comprehensive framework for developing, testing, and analyzing trading strategies with advanced performance metrics including Sharpe Ratio, Max Drawdown, Sortino Ratio, and Calmar Ratio.

## 🚀 Features

- **Advanced Performance Metrics**: Sharpe Ratio, Max Drawdown, Sortino Ratio, Calmar Ratio, VaR, CVaR
- **Multiple Strategy Types**: Momentum, RSI, Bollinger Bands, Custom strategies
- **Custom Data Format Support**: Handles specific tick data formats (id, price, qty, base_qty, time, is_buyer_maker)
- **Strategy Comparison**: Compare multiple strategies side-by-side
- **Comprehensive Analysis**: Win rate, profit factor, expectancy, and more
- **Export Capabilities**: CSV exports for further analysis

## 📊 Performance Metrics Included

- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Maximum peak-to-trough decline
- **Sortino Ratio**: Downside risk-adjusted returns
- **Calmar Ratio**: Return vs max drawdown
- **Value at Risk (VaR)**: 95% confidence level
- **Conditional VaR (CVaR)**: Expected loss beyond VaR
- **Information Ratio**: Active return vs tracking error
- **Treynor Ratio**: Return per unit of systematic risk
- **Jensen's Alpha**: Risk-adjusted excess return

## 🎯 Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib
```

### 2. Run Example
```bash
python enhanced_strategy_framework.py
```

### 3. Use Your Own Data
```python
from enhanced_strategy_framework import run_strategy_with_metrics, enhanced_momentum_strategy

# Load your data
tick_data = load_your_tick_data('your_data.csv')

# Run strategy with metrics
results = run_strategy_with_metrics(
    tick_data, 
    enhanced_momentum_strategy, 
    "My Strategy",
    {'lookback': 10, 'buy_threshold': 0.01}
)
```

## 📁 Core Files

- **`enhanced_strategy_framework.py`** - Main framework with multiple strategies
- **`advanced_performance_metrics.py`** - Performance calculations and analysis
- **`strategy_implementation_guide.py`** - How to add custom strategies
- **`custom_tick_data_strategy.py`** - Handles your specific data format

## 🔧 Adding Your Own Strategy

```python
def your_custom_strategy(tick_data, **strategy_params):
    """
    Add your custom trading logic here.
    """
    # Your strategy parameters
    lookback = strategy_params.get('lookback', 10)
    buy_threshold = strategy_params.get('buy_threshold', 0.01)
    
    # Your strategy logic
    # ... implement your signals and trading logic ...
    
    return trades, position, pnl
```

## 📈 Strategy Examples

### Momentum Strategy
- Uses price momentum with volume and volatility filters
- Configurable lookback periods and thresholds

### RSI Strategy
- Relative Strength Index based trading
- Oversold/overbought levels

### Bollinger Bands Strategy
- Mean reversion using Bollinger Bands
- Configurable periods and standard deviations

## 📊 Data Format

Your tick data should be in CSV format with columns:
- `id`: Trade ID
- `price`: Price
- `qty`: Quantity
- `base_qty`: Base quantity
- `time`: Timestamp (milliseconds)
- `is_buyer_maker`: Boolean flag

## 🎮 Usage Examples

### Basic Strategy Testing
```python
# Run single strategy
results = run_strategy_with_metrics(tick_data, enhanced_momentum_strategy, "Momentum")

# Access results
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {results['metrics']['max_drawdown_pct']:.2f}%")
```

### Strategy Comparison
```python
strategies = {
    'Momentum': (enhanced_momentum_strategy, {'lookback': 10}),
    'RSI': (rsi_strategy, {'rsi_period': 14}),
    'Bollinger': (bollinger_bands_strategy, {'period': 20})
}

results = compare_strategies(tick_data, strategies)
```

## 📋 Output Files

- **`strategy_comparison.csv`** - Comparison of all strategies
- **`metrics_[strategy_name].csv`** - Individual strategy metrics
- **`trades_[strategy_name].csv`** - Individual trade details

## 🔍 Performance Rating

The framework automatically rates strategies:
- **EXCELLENT ⭐⭐⭐⭐⭐**: Sharpe > 2, Max DD < 5%, Win Rate > 60%
- **GOOD ⭐⭐⭐⭐**: Sharpe > 1, Max DD < 10%, Win Rate > 50%
- **FAIR ⭐⭐⭐**: Moderate performance
- **POOR ⭐⭐**: Below average performance
- **VERY POOR ⭐**: Poor performance

## 🛠️ Customization

### Strategy Parameters
```python
strategy_params = {
    'lookback': 10,           # Lookback period
    'buy_threshold': 0.01,    # Buy signal threshold
    'sell_threshold': -0.01,  # Sell signal threshold
    'position_size': 0.1,     # Position size per trade
    'max_position': 1.0,      # Maximum position size
    'volume_filter': True,    # Enable volume filtering
    'volatility_filter': True # Enable volatility filtering
}
```

### Performance Metrics
```python
# Custom risk-free rate
metrics = calculate_advanced_metrics(trades, initial_capital=10000, risk_free_rate=0.03)

# Custom analysis
print_performance_report(metrics)
export_metrics_to_csv(metrics, "my_metrics.csv")
```

## 📚 Integration with ABIDES

This framework can be integrated with ABIDES for realistic market simulation:

1. **Quick Development**: Use this framework for rapid strategy development
2. **Advanced Testing**: Use ABIDES for realistic market simulation
3. **Production Ready**: Combine both for complete testing pipeline

## 🤝 Contributing

1. Fork the repository
2. Create your strategy branch
3. Add your custom strategy
4. Test with sample data
5. Submit pull request

## 📄 License

MIT License - Feel free to use and modify for your trading needs.

## 🆘 Support

For questions or issues:
1. Check the implementation guide
2. Review example strategies
3. Test with sample data first
4. Create issue with detailed description

---

**Happy Trading! 📈**
