#!/usr/bin/env python3
"""
Strategy Implementation Guide
Shows exactly where to add your custom strategy and how to get results.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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

# =============================================================================
# SECTION 1: WHERE TO ADD YOUR CUSTOM STRATEGY
# =============================================================================

def your_custom_strategy(tick_data, **strategy_params):
    """
    THIS IS WHERE YOU ADD YOUR CUSTOM STRATEGY!
    
    Replace this function with your own trading logic.
    
    Parameters:
    - tick_data: Your processed tick data
    - **strategy_params: Any custom parameters you want to pass
    
    Returns:
    - trades: List of trade dictionaries
    - final_position: Final position in base asset
    - final_pnl: Final P&L in quote currency
    """
    
    # =========================================================================
    # YOUR STRATEGY PARAMETERS (MODIFY THESE)
    # =========================================================================
    lookback = strategy_params.get('lookback', 10)
    buy_threshold = strategy_params.get('buy_threshold', 0.01)
    sell_threshold = strategy_params.get('sell_threshold', -0.01)
    position_size = strategy_params.get('position_size', 0.1)
    max_position = strategy_params.get('max_position', 1.0)
    
    # =========================================================================
    # YOUR STRATEGY LOGIC (REPLACE THIS WITH YOUR LOGIC)
    # =========================================================================
    
    # Initialize tracking variables
    position = 0.0
    pnl = 0.0
    trades = []
    price_history = []
    volume_history = []
    
    print(f"Running YOUR CUSTOM STRATEGY with parameters:")
    print(f"  Lookback: {lookback}")
    print(f"  Buy threshold: {buy_threshold}")
    print(f"  Sell threshold: {sell_threshold}")
    print(f"  Position size: {position_size}")
    print(f"  Max position: {max_position}")
    
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
        
        # =================================================================
        # YOUR SIGNAL CALCULATION (REPLACE THIS)
        # =================================================================
        
        if len(price_history) >= lookback:
            # Example: Simple momentum strategy
            recent_avg = np.mean(price_history[-lookback:-1])
            momentum_signal = (current_price - recent_avg) / recent_avg
            
            # Example: Volume-based signal
            recent_volume_avg = np.mean(volume_history[-lookback:-1])
            volume_signal = (current_volume - recent_volume_avg) / recent_volume_avg if recent_volume_avg > 0 else 0
            
            # Example: Combined signal (YOU CAN MODIFY THIS)
            combined_signal = momentum_signal + (volume_signal * 0.5)
            
            # =============================================================
            # YOUR TRADING LOGIC (REPLACE THIS)
            # =============================================================
            
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
                    'volume': volume_signal
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
                    'volume': volume_signal
                })
                print(f"SELL: {position_size} at ${current_price:.2f} (Signal: {combined_signal:.3f})")
    
    return trades, position, pnl

# =============================================================================
# SECTION 2: HOW TO GET RESULTS FROM YOUR STRATEGY
# =============================================================================

def analyze_strategy_results(trades, position, pnl, tick_data, strategy_name="Custom Strategy"):
    """
    THIS IS HOW YOU GET AND ANALYZE RESULTS FROM YOUR STRATEGY!
    
    This function shows you all the different ways to extract and analyze
    results from your trading strategy.
    """
    
    print(f"\n{'='*60}")
    print(f"RESULTS ANALYSIS: {strategy_name}")
    print(f"{'='*60}")
    
    # =========================================================================
    # BASIC RESULTS
    # =========================================================================
    print(f"📊 BASIC RESULTS:")
    print(f"  Final Position: {position:.4f} BTC")
    print(f"  Final P&L: ${pnl:.2f}")
    print(f"  Total Trades: {len(trades)}")
    
    if trades:
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        
        print(f"  Buy Trades: {len(buy_trades)}")
        print(f"  Sell Trades: {len(sell_trades)}")
        
        if buy_trades:
            avg_buy_price = np.mean([t['price'] for t in buy_trades])
            print(f"  Average Buy Price: ${avg_buy_price:.2f}")
        
        if sell_trades:
            avg_sell_price = np.mean([t['price'] for t in sell_trades])
            print(f"  Average Sell Price: ${avg_sell_price:.2f}")
    
    # =========================================================================
    # PERFORMANCE METRICS
    # =========================================================================
    print(f"\n📈 PERFORMANCE METRICS:")
    
    if trades and len(trades) > 1:
        # Calculate trade returns
        trade_returns = []
        for i in range(1, len(trades)):
            if trades[i]['action'] == 'SELL' and trades[i-1]['action'] == 'BUY':
                return_pct = (trades[i]['price'] - trades[i-1]['price']) / trades[i-1]['price']
                trade_returns.append(return_pct)
        
        if trade_returns:
            avg_return = np.mean(trade_returns)
            win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns)
            max_return = max(trade_returns)
            min_return = min(trade_returns)
            
            print(f"  Average Trade Return: {avg_return:.2%}")
            print(f"  Win Rate: {win_rate:.2%}")
            print(f"  Best Trade: {max_return:.2%}")
            print(f"  Worst Trade: {min_return:.2%}")
    
    # =========================================================================
    # SIGNAL ANALYSIS
    # =========================================================================
    print(f"\n🎯 SIGNAL ANALYSIS:")
    
    if trades:
        signals = [t['signal'] for t in trades]
        print(f"  Average Signal Strength: {np.mean(signals):.3f}")
        print(f"  Signal Range: {min(signals):.3f} to {max(signals):.3f}")
        print(f"  Signal Std Dev: {np.std(signals):.3f}")
    
    # =========================================================================
    # EXPORT RESULTS TO FILES
    # =========================================================================
    print(f"\n💾 EXPORTING RESULTS:")
    
    # Export trades to CSV
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_file = f"trades_{strategy_name.lower().replace(' ', '_')}.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"  Trades exported to: {trades_file}")
    
    # Export summary to CSV
    summary_data = {
        'strategy_name': [strategy_name],
        'final_position': [position],
        'final_pnl': [pnl],
        'total_trades': [len(trades)],
        'buy_trades': [len([t for t in trades if t['action'] == 'BUY'])],
        'sell_trades': [len([t for t in trades if t['action'] == 'SELL'])],
        'avg_signal': [np.mean([t['signal'] for t in trades]) if trades else 0],
        'data_ticks': [len(tick_data)],
        'price_range': [f"${tick_data['price'].min():.2f} - ${tick_data['price'].max():.2f}"]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = f"summary_{strategy_name.lower().replace(' ', '_')}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"  Summary exported to: {summary_file}")
    
    # =========================================================================
    # RETURN RESULTS FOR FURTHER ANALYSIS
    # =========================================================================
    results = {
        'trades': trades,
        'final_position': position,
        'final_pnl': pnl,
        'total_trades': len(trades),
        'buy_trades': len([t for t in trades if t['action'] == 'BUY']),
        'sell_trades': len([t for t in trades if t['action'] == 'SELL']),
        'avg_signal': np.mean([t['signal'] for t in trades]) if trades else 0,
        'data_ticks': len(tick_data)
    }
    
    return results

# =============================================================================
# SECTION 3: HOW TO RUN YOUR STRATEGY
# =============================================================================

def run_your_strategy(data_file, strategy_params=None):
    """
    THIS IS HOW YOU RUN YOUR STRATEGY!
    
    This function shows you the complete workflow:
    1. Load your data
    2. Run your strategy
    3. Get and analyze results
    """
    
    if strategy_params is None:
        strategy_params = {
            'lookback': 10,
            'buy_threshold': 0.01,
            'sell_threshold': -0.01,
            'position_size': 0.1,
            'max_position': 1.0
        }
    
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║ Running Your Custom Strategy                              ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    
    # Step 1: Load your data
    print("\n1. Loading your tick data...")
    tick_data = load_your_tick_data(data_file)
    print(f"   Loaded {len(tick_data)} ticks")
    
    # Step 2: Run your strategy
    print("\n2. Running your custom strategy...")
    trades, position, pnl = your_custom_strategy(tick_data, **strategy_params)
    
    # Step 3: Analyze results
    print("\n3. Analyzing results...")
    results = analyze_strategy_results(trades, position, pnl, tick_data, "Your Custom Strategy")
    
    # Step 4: Return results for further use
    print("\n4. Strategy completed!")
    return results

# =============================================================================
# SECTION 4: EXAMPLE USAGE
# =============================================================================

def main():
    """Example of how to use the strategy framework."""
    
    # Create sample data first
    print("Creating sample data...")
    num_ticks = 50
    base_price = 224.0
    
    prices = []
    current_price = base_price
    for i in range(num_ticks):
        change = np.random.normal(0, 0.3)
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
    
    sample_data.to_csv('sample_strategy_data.csv', index=False)
    print("Sample data created: sample_strategy_data.csv")
    
    # Example 1: Run with default parameters
    print("\n" + "="*60)
    print("EXAMPLE 1: Default Parameters")
    print("="*60)
    results1 = run_your_strategy('sample_strategy_data.csv')
    
    # Example 2: Run with custom parameters
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Parameters")
    print("="*60)
    custom_params = {
        'lookback': 5,
        'buy_threshold': 0.005,
        'sell_threshold': -0.005,
        'position_size': 0.05,
        'max_position': 0.5
    }
    results2 = run_your_strategy('sample_strategy_data.csv', custom_params)
    
    # Example 3: Compare results
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Default Strategy P&L: ${results1['final_pnl']:.2f}")
    print(f"Custom Strategy P&L: ${results2['final_pnl']:.2f}")
    
    print(f"\n{'='*60}")
    print("HOW TO USE WITH YOUR OWN DATA:")
    print(f"{'='*60}")
    print("1. Save your data as CSV with columns: id, price, qty, base_qty, time, is_buyer_maker")
    print("2. Modify the 'your_custom_strategy' function with your logic")
    print("3. Run: results = run_your_strategy('your_file.csv')")
    print("4. Access results: results['final_pnl'], results['trades'], etc.")

if __name__ == "__main__":
    main()
