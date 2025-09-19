#!/usr/bin/env python3
"""
Enhanced Strategy Framework with Advanced Performance Metrics
Integrates Sharpe Ratio, Max Drawdown, Sortino Ratio, Calmar Ratio, and more.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from advanced_performance_metrics import calculate_advanced_metrics, print_performance_report, export_metrics_to_csv

def load_your_tick_data(file_path):
    """Load your tick data format."""
    print(f"Loading tick data from {file_path}")
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
    
    tick_data = pd.DataFrame({
        'timestamp': df['timestamp'],
        'symbol': 'BTCUSDT',
        'price': df['price'],
        'volume': df['qty'],
        'base_volume': df['base_qty'],
        'is_buyer_maker': df['is_buyer_maker'],
        'trade_id': df['id']
    })
    
    return tick_data.sort_values('timestamp')

def enhanced_momentum_strategy(tick_data, **strategy_params):
    """
    Enhanced momentum strategy with advanced signal generation.
    """
    # Strategy parameters
    lookback = strategy_params.get('lookback', 10)
    buy_threshold = strategy_params.get('buy_threshold', 0.01)
    sell_threshold = strategy_params.get('sell_threshold', -0.01)
    position_size = strategy_params.get('position_size', 0.1)
    max_position = strategy_params.get('max_position', 1.0)
    volume_filter = strategy_params.get('volume_filter', True)
    volatility_filter = strategy_params.get('volatility_filter', True)
    
    # Initialize tracking variables
    position = 0.0
    pnl = 0.0
    trades = []
    price_history = []
    volume_history = []
    
    print(f"Running ENHANCED MOMENTUM STRATEGY with parameters:")
    print(f"  Lookback: {lookback}")
    print(f"  Buy threshold: {buy_threshold}")
    print(f"  Sell threshold: {sell_threshold}")
    print(f"  Position size: {position_size}")
    print(f"  Max position: {max_position}")
    print(f"  Volume filter: {volume_filter}")
    print(f"  Volatility filter: {volatility_filter}")
    
    for i, row in tick_data.iterrows():
        current_price = row['price']
        current_volume = row['volume']
        is_buyer_maker = row['is_buyer_maker']
        
        # Update history
        price_history.append(current_price)
        volume_history.append(current_volume)
        
        # Keep only recent history
        if len(price_history) > lookback * 2:
            price_history = price_history[-lookback:]
            volume_history = volume_history[-lookback:]
        
        if len(price_history) >= lookback:
            # Calculate momentum signal
            recent_avg = np.mean(price_history[-lookback:-1])
            momentum_signal = (current_price - recent_avg) / recent_avg
            
            # Calculate volume signal
            recent_volume_avg = np.mean(volume_history[-lookback:-1])
            volume_signal = (current_volume - recent_volume_avg) / recent_volume_avg if recent_volume_avg > 0 else 0
            
            # Calculate volatility signal
            price_std = np.std(price_history[-lookback:])
            volatility_signal = price_std / np.mean(price_history[-lookback:])
            
            # Combined signal with filters
            combined_signal = momentum_signal
            
            # Apply volume filter
            if volume_filter and volume_signal < 0.5:
                combined_signal *= 0.5  # Reduce signal strength for low volume
            
            # Apply volatility filter
            if volatility_filter and volatility_signal > 0.05:
                combined_signal *= 0.7  # Reduce signal strength for high volatility
            
            # Trading logic
            if combined_signal > buy_threshold and position < max_position:
                # Buy signal
                position += position_size
                pnl -= current_price * position_size
                trades.append({
                    'time': row['timestamp'],
                    'action': 'BUY',
                    'price': current_price,
                    'size': position_size,
                    'position': position,
                    'pnl': pnl,
                    'signal': combined_signal,
                    'momentum': momentum_signal,
                    'volume': volume_signal,
                    'volatility': volatility_signal
                })
                print(f"BUY: {position_size} at ${current_price:.2f} (Signal: {combined_signal:.3f})")
                
            elif combined_signal < sell_threshold and position > -max_position:
                # Sell signal
                position -= position_size
                pnl += current_price * position_size
                trades.append({
                    'time': row['timestamp'],
                    'action': 'SELL',
                    'price': current_price,
                    'size': position_size,
                    'position': position,
                    'pnl': pnl,
                    'signal': combined_signal,
                    'momentum': momentum_signal,
                    'volume': volume_signal,
                    'volatility': volatility_signal
                })
                print(f"SELL: {position_size} at ${current_price:.2f} (Signal: {combined_signal:.3f})")
    
    return trades, position, pnl

def rsi_strategy(tick_data, **strategy_params):
    """
    RSI-based trading strategy.
    """
    rsi_period = strategy_params.get('rsi_period', 14)
    oversold = strategy_params.get('oversold', 30)
    overbought = strategy_params.get('overbought', 70)
    position_size = strategy_params.get('position_size', 0.1)
    max_position = strategy_params.get('max_position', 1.0)
    
    position = 0.0
    pnl = 0.0
    trades = []
    prices = []
    
    print(f"Running RSI STRATEGY with parameters:")
    print(f"  RSI Period: {rsi_period}")
    print(f"  Oversold: {oversold}")
    print(f"  Overbought: {overbought}")
    print(f"  Position size: {position_size}")
    
    for i, row in tick_data.iterrows():
        prices.append(row['price'])
        
        if len(prices) >= rsi_period + 1:
            # Calculate RSI
            rsi = calculate_rsi(prices[-rsi_period-1:])
            
            # Trading logic
            if rsi < oversold and position < max_position:
                # Buy signal
                position += position_size
                pnl -= row['price'] * position_size
                trades.append({
                    'time': row['timestamp'],
                    'action': 'BUY',
                    'price': row['price'],
                    'size': position_size,
                    'position': position,
                    'pnl': pnl,
                    'rsi': rsi
                })
                print(f"BUY: {position_size} at ${row['price']:.2f} (RSI: {rsi:.1f})")
                
            elif rsi > overbought and position > -max_position:
                # Sell signal
                position -= position_size
                pnl += row['price'] * position_size
                trades.append({
                    'time': row['timestamp'],
                    'action': 'SELL',
                    'price': row['price'],
                    'size': position_size,
                    'position': position,
                    'pnl': pnl,
                    'rsi': rsi
                })
                print(f"SELL: {position_size} at ${row['price']:.2f} (RSI: {rsi:.1f})")
    
    return trades, position, pnl

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    if len(prices) < period + 1:
        return 50
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def bollinger_bands_strategy(tick_data, **strategy_params):
    """
    Bollinger Bands trading strategy.
    """
    period = strategy_params.get('period', 20)
    std_dev = strategy_params.get('std_dev', 2)
    position_size = strategy_params.get('position_size', 0.1)
    max_position = strategy_params.get('max_position', 1.0)
    
    position = 0.0
    pnl = 0.0
    trades = []
    prices = []
    
    print(f"Running BOLLINGER BANDS STRATEGY with parameters:")
    print(f"  Period: {period}")
    print(f"  Std Dev: {std_dev}")
    print(f"  Position size: {position_size}")
    
    for i, row in tick_data.iterrows():
        prices.append(row['price'])
        
        if len(prices) >= period:
            # Calculate Bollinger Bands
            sma = np.mean(prices[-period:])
            std = np.std(prices[-period:])
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            current_price = row['price']
            
            # Trading logic
            if current_price <= lower_band and position < max_position:
                # Buy signal (price at lower band)
                position += position_size
                pnl -= current_price * position_size
                trades.append({
                    'time': row['timestamp'],
                    'action': 'BUY',
                    'price': current_price,
                    'size': position_size,
                    'position': position,
                    'pnl': pnl,
                    'upper_band': upper_band,
                    'lower_band': lower_band,
                    'sma': sma
                })
                print(f"BUY: {position_size} at ${current_price:.2f} (Lower Band: ${lower_band:.2f})")
                
            elif current_price >= upper_band and position > -max_position:
                # Sell signal (price at upper band)
                position -= position_size
                pnl += current_price * position_size
                trades.append({
                    'time': row['timestamp'],
                    'action': 'SELL',
                    'price': current_price,
                    'size': position_size,
                    'position': position,
                    'pnl': pnl,
                    'upper_band': upper_band,
                    'lower_band': lower_band,
                    'sma': sma
                })
                print(f"SELL: {position_size} at ${current_price:.2f} (Upper Band: ${upper_band:.2f})")
    
    return trades, position, pnl

def run_strategy_with_metrics(tick_data, strategy_func, strategy_name, strategy_params=None, initial_capital=10000):
    """
    Run a strategy and calculate comprehensive performance metrics.
    """
    if strategy_params is None:
        strategy_params = {}
    
    print(f"\n{'='*80}")
    print(f"RUNNING STRATEGY: {strategy_name}")
    print(f"{'='*80}")
    
    # Run the strategy
    trades, position, pnl = strategy_func(tick_data, **strategy_params)
    
    # Calculate advanced metrics
    metrics = calculate_advanced_metrics(trades, initial_capital)
    
    # Print performance report
    print_performance_report(metrics)
    
    # Export metrics
    filename = f"metrics_{strategy_name.lower().replace(' ', '_')}.csv"
    export_metrics_to_csv(metrics, filename)
    
    return {
        'strategy_name': strategy_name,
        'trades': trades,
        'position': position,
        'pnl': pnl,
        'metrics': metrics
    }

def compare_strategies(tick_data, strategies, initial_capital=10000):
    """
    Compare multiple strategies and their performance metrics.
    """
    results = []
    
    print(f"\n{'='*80}")
    print(f"STRATEGY COMPARISON")
    print(f"{'='*80}")
    
    for strategy_name, (strategy_func, params) in strategies.items():
        result = run_strategy_with_metrics(tick_data, strategy_func, strategy_name, params, initial_capital)
        results.append(result)
    
    # Create comparison table
    print(f"\n{'='*80}")
    print(f"STRATEGY COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    comparison_data = []
    for result in results:
        metrics = result['metrics']
        comparison_data.append({
            'Strategy': result['strategy_name'],
            'Total Return (%)': f"{metrics['total_return_pct']:.2f}",
            'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
            'Max Drawdown (%)': f"{metrics['max_drawdown_pct']:.2f}",
            'Calmar Ratio': f"{metrics['calmar_ratio']:.3f}",
            'Sortino Ratio': f"{metrics['sortino_ratio']:.3f}",
            'Win Rate (%)': f"{metrics['win_rate']:.1f}",
            'Total Trades': metrics['total_trades']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Export comparison
    comparison_df.to_csv('strategy_comparison.csv', index=False)
    print(f"\n💾 Strategy comparison exported to: strategy_comparison.csv")
    
    return results

def main():
    """Main function to demonstrate the enhanced framework."""
    
    # Create sample data
    print("Creating sample data...")
    num_ticks = 100
    base_price = 224.0
    
    prices = []
    current_price = base_price
    for i in range(num_ticks):
        change = np.random.normal(0, 0.5)
        current_price += change
        prices.append(round(current_price, 2))
    
    ids = list(range(313571255, 313571255 + num_ticks))
    qtys = np.random.uniform(0.01, 1.5, num_ticks)
    base_qtys = [qty * price for qty, price in zip(qtys, prices)]
    times = [1757550000000 + i * 1000 for i in range(num_ticks)]
    is_buyer_makers = np.random.choice([True, False], num_ticks, p=[0.6, 0.4])
    
    sample_data = pd.DataFrame({
        'id': ids,
        'price': prices,
        'qty': qtys,
        'base_qty': base_qtys,
        'time': times,
        'is_buyer_maker': is_buyer_makers
    })
    
    sample_data.to_csv('enhanced_sample_data.csv', index=False)
    print("Sample data created: enhanced_sample_data.csv")
    
    # Load data
    tick_data = load_your_tick_data('enhanced_sample_data.csv')
    
    # Define strategies to compare
    strategies = {
        'Enhanced Momentum': (enhanced_momentum_strategy, {
            'lookback': 10,
            'buy_threshold': 0.01,
            'sell_threshold': -0.01,
            'position_size': 0.1,
            'volume_filter': True,
            'volatility_filter': True
        }),
        'RSI Strategy': (rsi_strategy, {
            'rsi_period': 14,
            'oversold': 30,
            'overbought': 70,
            'position_size': 0.1
        }),
        'Bollinger Bands': (bollinger_bands_strategy, {
            'period': 20,
            'std_dev': 2,
            'position_size': 0.1
        })
    }
    
    # Compare strategies
    results = compare_strategies(tick_data, strategies, initial_capital=10000)
    
    print(f"\n{'='*80}")
    print(f"BEST PERFORMING STRATEGY")
    print(f"{'='*80}")
    
    # Find best strategy by Sharpe ratio
    best_strategy = max(results, key=lambda x: x['metrics']['sharpe_ratio'])
    print(f"Best Strategy: {best_strategy['strategy_name']}")
    print(f"Sharpe Ratio: {best_strategy['metrics']['sharpe_ratio']:.3f}")
    print(f"Total Return: {best_strategy['metrics']['total_return_pct']:.2f}%")
    print(f"Max Drawdown: {best_strategy['metrics']['max_drawdown_pct']:.2f}%")

if __name__ == "__main__":
    main()
