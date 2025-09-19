#!/usr/bin/env python3
"""
Advanced Performance Metrics for Trading Strategies
Calculates Sharpe Ratio, Max Drawdown, Sortino Ratio, Calmar Ratio, and more.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def calculate_advanced_metrics(trades, initial_capital=10000, risk_free_rate=0.02):
    """
    Calculate comprehensive performance metrics for trading strategies.
    
    Parameters:
    - trades: List of trade dictionaries
    - initial_capital: Starting capital
    - risk_free_rate: Annual risk-free rate (default 2%)
    
    Returns:
    - Dictionary with all performance metrics
    """
    
    if not trades:
        return create_empty_metrics()
    
    # Convert trades to DataFrame for easier analysis
    trades_df = pd.DataFrame(trades)
    trades_df['time'] = pd.to_datetime(trades_df['time'])
    trades_df = trades_df.sort_values('time')
    
    # Calculate portfolio value over time
    portfolio_values = calculate_portfolio_values(trades_df, initial_capital)
    
    # Calculate returns
    returns = calculate_returns(portfolio_values)
    
    # Calculate all metrics
    metrics = {}
    
    # Basic metrics
    metrics.update(calculate_basic_metrics(trades_df, portfolio_values, initial_capital))
    
    # Risk metrics
    metrics.update(calculate_risk_metrics(returns, risk_free_rate))
    
    # Drawdown metrics
    metrics.update(calculate_drawdown_metrics(portfolio_values))
    
    # Advanced ratios
    metrics.update(calculate_advanced_ratios(returns, risk_free_rate, metrics))
    
    # Trade analysis
    metrics.update(analyze_trades(trades_df))
    
    return metrics

def calculate_portfolio_values(trades_df, initial_capital):
    """Calculate portfolio value over time."""
    portfolio_values = []
    current_value = initial_capital
    
    for _, trade in trades_df.iterrows():
        # Update portfolio value based on trade
        if trade['action'] == 'BUY':
            current_value -= trade['price'] * trade['size']
        else:  # SELL
            current_value += trade['price'] * trade['size']
        
        # Add unrealized P&L from current position
        if 'position' in trade and trade['position'] != 0:
            current_value += trade['position'] * trade['price']
        
        portfolio_values.append({
            'time': trade['time'],
            'value': current_value,
            'trade_pnl': trade.get('pnl', 0)
        })
    
    return pd.DataFrame(portfolio_values)

def calculate_returns(portfolio_values):
    """Calculate returns from portfolio values."""
    if len(portfolio_values) < 2:
        return pd.Series([0])
    
    values = portfolio_values['value'].values
    returns = np.diff(values) / values[:-1]
    return pd.Series(returns)

def calculate_basic_metrics(trades_df, portfolio_values, initial_capital):
    """Calculate basic performance metrics."""
    final_value = portfolio_values['value'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    buy_trades = trades_df[trades_df['action'] == 'BUY']
    sell_trades = trades_df[trades_df['action'] == 'SELL']
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'total_trades': len(trades_df),
        'buy_trades': len(buy_trades),
        'sell_trades': len(sell_trades),
        'win_rate': calculate_win_rate(trades_df)
    }

def calculate_win_rate(trades_df):
    """Calculate win rate from trades."""
    if len(trades_df) < 2:
        return 0
    
    # Calculate trade returns
    trade_returns = []
    for i in range(1, len(trades_df)):
        if (trades_df.iloc[i]['action'] == 'SELL' and 
            trades_df.iloc[i-1]['action'] == 'BUY'):
            buy_price = trades_df.iloc[i-1]['price']
            sell_price = trades_df.iloc[i]['price']
            trade_return = (sell_price - buy_price) / buy_price
            trade_returns.append(trade_return)
    
    if not trade_returns:
        return 0
    
    winning_trades = len([r for r in trade_returns if r > 0])
    return winning_trades / len(trade_returns)

def calculate_risk_metrics(returns, risk_free_rate):
    """Calculate risk metrics."""
    if len(returns) == 0:
        return {
            'volatility': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'var_95': 0,
            'cvar_95': 0
        }
    
    # Annualize returns (assuming daily data)
    annualized_returns = returns.mean() * 252
    annualized_volatility = returns.std() * np.sqrt(252)
    
    # Risk-free rate (annualized)
    rf_rate = risk_free_rate
    
    # Sharpe Ratio
    sharpe_ratio = (annualized_returns - rf_rate) / annualized_volatility if annualized_volatility > 0 else 0
    
    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = (annualized_returns - rf_rate) / downside_volatility if downside_volatility > 0 else 0
    
    # Value at Risk (95%)
    var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
    
    # Conditional Value at Risk (95%)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
    
    return {
        'volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'annualized_return': annualized_returns
    }

def calculate_drawdown_metrics(portfolio_values):
    """Calculate drawdown metrics."""
    if len(portfolio_values) == 0:
        return {
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'calmar_ratio': 0,
            'recovery_time': 0
        }
    
    values = portfolio_values['value'].values
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    
    max_drawdown = np.min(drawdown)
    max_drawdown_pct = max_drawdown * 100
    
    # Calmar Ratio (Annual Return / Max Drawdown)
    if len(values) > 1:
        total_return = (values[-1] - values[0]) / values[0]
        annualized_return = (1 + total_return) ** (252 / len(values)) - 1
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    else:
        calmar_ratio = 0
    
    # Recovery time (simplified)
    recovery_time = calculate_recovery_time(drawdown)
    
    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'calmar_ratio': calmar_ratio,
        'recovery_time': recovery_time
    }

def calculate_recovery_time(drawdown):
    """Calculate average recovery time from drawdowns."""
    recovery_times = []
    in_drawdown = False
    drawdown_start = 0
    
    for i, dd in enumerate(drawdown):
        if dd < 0 and not in_drawdown:
            in_drawdown = True
            drawdown_start = i
        elif dd >= 0 and in_drawdown:
            in_drawdown = False
            recovery_times.append(i - drawdown_start)
    
    return np.mean(recovery_times) if recovery_times else 0

def calculate_advanced_ratios(returns, risk_free_rate, metrics):
    """Calculate advanced performance ratios."""
    if len(returns) == 0:
        return {
            'information_ratio': 0,
            'treynor_ratio': 0,
            'jensen_alpha': 0
        }
    
    # Information Ratio (simplified - assumes benchmark return of 0)
    benchmark_return = 0
    excess_returns = returns - benchmark_return
    tracking_error = excess_returns.std() * np.sqrt(252)
    information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
    
    # Treynor Ratio (simplified - assumes beta of 1)
    beta = 1.0  # Simplified assumption
    treynor_ratio = (returns.mean() * 252 - risk_free_rate) / beta if beta != 0 else 0
    
    # Jensen's Alpha (simplified)
    market_return = risk_free_rate + 0.08  # Assume 8% market return
    jensen_alpha = returns.mean() * 252 - (risk_free_rate + beta * (market_return - risk_free_rate))
    
    return {
        'information_ratio': information_ratio,
        'treynor_ratio': treynor_ratio,
        'jensen_alpha': jensen_alpha
    }

def analyze_trades(trades_df):
    """Analyze individual trade performance."""
    if len(trades_df) == 0:
        return {
            'avg_trade_return': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'profit_factor': 0,
            'expectancy': 0
        }
    
    # Calculate trade returns
    trade_returns = []
    for i in range(1, len(trades_df)):
        if (trades_df.iloc[i]['action'] == 'SELL' and 
            trades_df.iloc[i-1]['action'] == 'BUY'):
            buy_price = trades_df.iloc[i-1]['price']
            sell_price = trades_df.iloc[i]['price']
            trade_return = (sell_price - buy_price) / buy_price
            trade_returns.append(trade_return)
    
    if not trade_returns:
        return {
            'avg_trade_return': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'profit_factor': 0,
            'expectancy': 0
        }
    
    winning_trades = [r for r in trade_returns if r > 0]
    losing_trades = [r for r in trade_returns if r < 0]
    
    avg_trade_return = np.mean(trade_returns)
    best_trade = max(trade_returns)
    worst_trade = min(trade_returns)
    
    # Profit Factor
    gross_profit = sum(winning_trades) if winning_trades else 0
    gross_loss = abs(sum(losing_trades)) if losing_trades else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Expectancy
    expectancy = avg_trade_return
    
    return {
        'avg_trade_return': avg_trade_return,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'profit_factor': profit_factor,
        'expectancy': expectancy
    }

def create_empty_metrics():
    """Create empty metrics dictionary."""
    return {
        'initial_capital': 0,
        'final_value': 0,
        'total_return': 0,
        'total_return_pct': 0,
        'total_trades': 0,
        'buy_trades': 0,
        'sell_trades': 0,
        'win_rate': 0,
        'volatility': 0,
        'sharpe_ratio': 0,
        'sortino_ratio': 0,
        'var_95': 0,
        'cvar_95': 0,
        'annualized_return': 0,
        'max_drawdown': 0,
        'max_drawdown_pct': 0,
        'calmar_ratio': 0,
        'recovery_time': 0,
        'information_ratio': 0,
        'treynor_ratio': 0,
        'jensen_alpha': 0,
        'avg_trade_return': 0,
        'best_trade': 0,
        'worst_trade': 0,
        'profit_factor': 0,
        'expectancy': 0
    }

def print_performance_report(metrics):
    """Print a comprehensive performance report."""
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE PERFORMANCE REPORT")
    print(f"{'='*80}")
    
    # Basic Performance
    print(f"\n📊 BASIC PERFORMANCE:")
    print(f"  Initial Capital: ${metrics['initial_capital']:,.2f}")
    print(f"  Final Value: ${metrics['final_value']:,.2f}")
    print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"  Annualized Return: {metrics['annualized_return']:.2f}%")
    
    # Risk Metrics
    print(f"\n⚠️  RISK METRICS:")
    print(f"  Volatility: {metrics['volatility']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    print(f"  VaR (95%): {metrics['var_95']:.3f}")
    print(f"  CVaR (95%): {metrics['cvar_95']:.3f}")
    
    # Advanced Ratios
    print(f"\n🎯 ADVANCED RATIOS:")
    print(f"  Information Ratio: {metrics['information_ratio']:.3f}")
    print(f"  Treynor Ratio: {metrics['treynor_ratio']:.3f}")
    print(f"  Jensen's Alpha: {metrics['jensen_alpha']:.3f}")
    
    # Trade Analysis
    print(f"\n📈 TRADE ANALYSIS:")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2f}%")
    print(f"  Average Trade Return: {metrics['avg_trade_return']:.3f}")
    print(f"  Best Trade: {metrics['best_trade']:.3f}")
    print(f"  Worst Trade: {metrics['worst_trade']:.3f}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Expectancy: {metrics['expectancy']:.3f}")
    
    # Performance Rating
    print(f"\n🏆 PERFORMANCE RATING:")
    rating = rate_performance(metrics)
    print(f"  Overall Rating: {rating}")

def rate_performance(metrics):
    """Rate performance based on key metrics."""
    score = 0
    
    # Sharpe Ratio scoring
    if metrics['sharpe_ratio'] > 2:
        score += 3
    elif metrics['sharpe_ratio'] > 1:
        score += 2
    elif metrics['sharpe_ratio'] > 0.5:
        score += 1
    
    # Max Drawdown scoring
    if metrics['max_drawdown_pct'] > -5:
        score += 3
    elif metrics['max_drawdown_pct'] > -10:
        score += 2
    elif metrics['max_drawdown_pct'] > -20:
        score += 1
    
    # Win Rate scoring
    if metrics['win_rate'] > 60:
        score += 2
    elif metrics['win_rate'] > 50:
        score += 1
    
    # Calmar Ratio scoring
    if metrics['calmar_ratio'] > 1:
        score += 2
    elif metrics['calmar_ratio'] > 0.5:
        score += 1
    
    if score >= 8:
        return "EXCELLENT ⭐⭐⭐⭐⭐"
    elif score >= 6:
        return "GOOD ⭐⭐⭐⭐"
    elif score >= 4:
        return "FAIR ⭐⭐⭐"
    elif score >= 2:
        return "POOR ⭐⭐"
    else:
        return "VERY POOR ⭐"

def export_metrics_to_csv(metrics, filename="performance_metrics.csv"):
    """Export metrics to CSV file."""
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(filename, index=False)
    print(f"\n💾 Performance metrics exported to: {filename}")

# Example usage
if __name__ == "__main__":
    # Create sample trades for testing
    sample_trades = [
        {'time': '2025-01-01 09:00:00', 'action': 'BUY', 'price': 100, 'size': 1, 'position': 1, 'pnl': -100},
        {'time': '2025-01-01 10:00:00', 'action': 'SELL', 'price': 105, 'size': 1, 'position': 0, 'pnl': 5},
        {'time': '2025-01-01 11:00:00', 'action': 'BUY', 'price': 103, 'size': 1, 'position': 1, 'pnl': -98},
        {'time': '2025-01-01 12:00:00', 'action': 'SELL', 'price': 108, 'size': 1, 'position': 0, 'pnl': 10},
    ]
    
    # Calculate metrics
    metrics = calculate_advanced_metrics(sample_trades, initial_capital=10000)
    
    # Print report
    print_performance_report(metrics)
    
    # Export to CSV
    export_metrics_to_csv(metrics)
