#!/usr/bin/env python3
"""
Custom Tick Data Strategy for Your Data Format
Handles the specific format: id, price, qty, base_qty, time, is_buyer_maker
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_your_tick_data(file_path):
    """Load and process your specific tick data format."""
    print(f"Loading tick data from {file_path}")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Convert timestamp from scientific notation to datetime
    # Assuming the time column is in milliseconds
    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
    
    # Create a clean dataset for trading
    tick_data = pd.DataFrame({
        'timestamp': df['timestamp'],
        'symbol': 'BTCUSDT',  # Assuming crypto data, adjust as needed
        'price': df['price'],
        'volume': df['qty'],
        'base_volume': df['base_qty'],
        'is_buyer_maker': df['is_buyer_maker'],
        'trade_id': df['id']
    })
    
    # Sort by timestamp
    tick_data = tick_data.sort_values('timestamp')
    
    print(f"\nProcessed data shape: {tick_data.shape}")
    print(f"Date range: {tick_data['timestamp'].min()} to {tick_data['timestamp'].max()}")
    print(f"Price range: ${tick_data['price'].min():.2f} to ${tick_data['price'].max():.2f}")
    
    return tick_data

def advanced_trading_strategy(tick_data, strategy_type="momentum"):
    """Advanced trading strategy that uses your data format."""
    print(f"\nRunning {strategy_type} trading strategy...")
    
    # Strategy parameters
    lookback = 10
    buy_threshold = 0.005  # 0.5%
    sell_threshold = -0.005  # -0.5%
    position_size = 0.1  # BTC amount per trade
    
    # Initialize
    position = 0.0  # BTC position
    usd_pnl = 0.0   # USD P&L
    trades = []
    price_history = []
    volume_history = []
    maker_buy_ratio = []
    
    print("Processing ticks...")
    
    for i, row in tick_data.iterrows():
        current_price = row['price']
        current_volume = row['volume']
        is_buyer_maker = row['is_buyer_maker']
        
        # Update history
        price_history.append(current_price)
        volume_history.append(current_volume)
        maker_buy_ratio.append(1 if is_buyer_maker else 0)
        
        # Keep only recent history
        if len(price_history) > lookback * 2:
            price_history = price_history[-lookback:]
            volume_history = volume_history[-lookback:]
            maker_buy_ratio = maker_buy_ratio[-lookback:]
        
        # Calculate signals
        if len(price_history) >= lookback:
            # Price momentum signal
            recent_avg = np.mean(price_history[-lookback:-1])
            price_momentum = (current_price - recent_avg) / recent_avg
            
            # Volume signal
            recent_volume_avg = np.mean(volume_history[-lookback:-1])
            volume_signal = (current_volume - recent_volume_avg) / recent_volume_avg if recent_volume_avg > 0 else 0
            
            # Market maker signal (buyer maker ratio)
            maker_ratio = np.mean(maker_buy_ratio[-lookback:])
            
            # Combined signal based on strategy type
            if strategy_type == "momentum":
                signal = price_momentum
            elif strategy_type == "volume":
                signal = volume_signal
            elif strategy_type == "market_microstructure":
                # Use maker/taker information
                signal = price_momentum * (1 - maker_ratio)  # Less momentum when more makers
            else:
                signal = price_momentum
            
            # Trading logic
            if signal > buy_threshold and position < 1.0:  # Max 1 BTC position
                # Buy
                position += position_size
                usd_pnl -= current_price * position_size
                trades.append({
                    'time': row['timestamp'],
                    'action': 'BUY',
                    'price': current_price,
                    'size': position_size,
                    'position': position,
                    'usd_pnl': usd_pnl,
                    'signal': signal,
                    'volume_signal': volume_signal,
                    'maker_ratio': maker_ratio
                })
                print(f"BUY: {position_size} BTC at ${current_price:.2f} (Signal: {signal:.3f}, Maker: {maker_ratio:.2f})")
                
            elif signal < sell_threshold and position > -1.0:  # Max -1 BTC position
                # Sell
                position -= position_size
                usd_pnl += current_price * position_size
                trades.append({
                    'time': row['timestamp'],
                    'action': 'SELL',
                    'price': current_price,
                    'size': position_size,
                    'position': position,
                    'usd_pnl': usd_pnl,
                    'signal': signal,
                    'volume_signal': volume_signal,
                    'maker_ratio': maker_ratio
                })
                print(f"SELL: {position_size} BTC at ${current_price:.2f} (Signal: {signal:.3f}, Maker: {maker_ratio:.2f})")
    
    return trades, position, usd_pnl

def analyze_custom_results(trades, position, usd_pnl, tick_data):
    """Analyze and display results for your custom data."""
    print(f"\n{'='*60}")
    print("CUSTOM TICK DATA TRADING RESULTS")
    print(f"{'='*60}")
    
    print(f"Final Position: {position:.4f} BTC")
    print(f"Final USD P&L: ${usd_pnl:.2f}")
    print(f"Total Trades: {len(trades)}")
    
    if trades:
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        
        print(f"Buy Trades: {len(buy_trades)}")
        print(f"Sell Trades: {len(sell_trades)}")
        
        if buy_trades:
            avg_buy_price = np.mean([t['price'] for t in buy_trades])
            print(f"Average Buy Price: ${avg_buy_price:.2f}")
        
        if sell_trades:
            avg_sell_price = np.mean([t['price'] for t in sell_trades])
            print(f"Average Sell Price: ${avg_sell_price:.2f}")
        
        # Calculate performance metrics
        if len(trades) > 1:
            trade_returns = []
            for i in range(1, len(trades)):
                if trades[i]['action'] == 'SELL' and trades[i-1]['action'] == 'BUY':
                    return_pct = (trades[i]['price'] - trades[i-1]['price']) / trades[i-1]['price']
                    trade_returns.append(return_pct)
            
            if trade_returns:
                avg_return = np.mean(trade_returns)
                win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns)
                print(f"Average Trade Return: {avg_return:.2%}")
                print(f"Win Rate: {win_rate:.2%}")
    
    # Data statistics
    print(f"\nData Statistics:")
    print(f"Total Ticks: {len(tick_data)}")
    print(f"Price Range: ${tick_data['price'].min():.2f} - ${tick_data['price'].max():.2f}")
    print(f"Volume Range: {tick_data['volume'].min():.4f} - {tick_data['volume'].max():.4f}")
    print(f"Maker Ratio: {tick_data['is_buyer_maker'].mean():.2%}")
    
    if trades:
        print(f"\nRecent Trades:")
        for i, trade in enumerate(trades[-5:], 1):  # Show last 5 trades
            print(f"  {i}. {trade['action']}: {trade['size']} BTC at ${trade['price']:.2f} "
                  f"(Signal: {trade['signal']:.3f}, P&L: ${trade['usd_pnl']:.2f})")

def create_sample_data_your_format():
    """Create sample data in your exact format."""
    print("Creating sample data in your format...")
    
    # Create sample data
    num_ticks = 100
    base_price = 224.0
    
    # Generate realistic price movements
    prices = []
    current_price = base_price
    for i in range(num_ticks):
        change = np.random.normal(0, 0.5)
        current_price += change
        prices.append(round(current_price, 2))
    
    # Generate other fields
    ids = list(range(313571255, 313571255 + num_ticks))
    qtys = np.random.uniform(0.01, 2.0, num_ticks)
    base_qtys = [qty * price for qty, price in zip(qtys, prices)]
    times = [1757550000000 + i * 1000 for i in range(num_ticks)]  # Milliseconds
    is_buyer_makers = np.random.choice([True, False], num_ticks, p=[0.6, 0.4])
    
    sample_data = pd.DataFrame({
        'id': ids,
        'price': prices,
        'qty': qtys,
        'base_qty': base_qtys,
        'time': times,
        'is_buyer_maker': is_buyer_makers
    })
    
    # Save sample data
    sample_data.to_csv('sample_your_format.csv', index=False)
    print(f"Created {len(sample_data)} sample records in your format")
    print("Saved to 'sample_your_format.csv'")
    
    return sample_data

def main():
    """Main function."""
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║ Custom Tick Data Strategy for Your Format                ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    
    # Check if you have your own data file
    your_data_file = "your_tick_data.csv"  # Replace with your actual file name
    
    if input("Do you have your own tick data file? (y/n): ").lower() == 'y':
        file_path = input("Enter the path to your CSV file: ")
        try:
            tick_data = load_your_tick_data(file_path)
        except Exception as e:
            print(f"Error loading your data: {e}")
            print("Using sample data instead...")
            create_sample_data_your_format()
            tick_data = load_your_tick_data('sample_your_format.csv')
    else:
        # Create and use sample data
        create_sample_data_your_format()
        tick_data = load_your_tick_data('sample_your_format.csv')
    
    # Run different strategies
    strategies = ["momentum", "volume", "market_microstructure"]
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"RUNNING {strategy.upper()} STRATEGY")
        print(f"{'='*60}")
        
        trades, position, usd_pnl = advanced_trading_strategy(tick_data, strategy)
        analyze_custom_results(trades, position, usd_pnl, tick_data)
    
    print(f"\n{'='*60}")
    print("HOW TO USE YOUR OWN DATA:")
    print(f"{'='*60}")
    print("1. Save your data as CSV with columns: id, price, qty, base_qty, time, is_buyer_maker")
    print("2. Run: python custom_tick_data_strategy.py")
    print("3. Enter the path to your CSV file when prompted")
    print("4. The script will automatically detect and process your format")

if __name__ == "__main__":
    main()
