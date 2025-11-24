#!/usr/bin/env python3
"""
自适应优化版ETH交易机器人
根据当前市场条件调整参数
"""

import ccxt
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('adaptive_optimized_scalper.log'),
        logging.StreamHandler()
    ]
)

class AdaptiveOptimizedScalper:
    """自适应优化版ETH交易机器人"""

    def __init__(self, initial_balance=10000):
        # 基础优化参数（基于最佳回测结果）
        self.base_params = {
            'rsi_period': 7,
            'rsi_oversold': 25,
            'rsi_overbought': 70,
            'ma_short': 5,
            'ma_long': 21,
            'bb_period': 15,
            'bb_std': 1.8,
            'min_signal_strength': 0.4,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'risk_per_trade': 0.01,
            'min_time_between_trades': 300,
            'max_holding_time': 3600
        }

        # 自适应参数
        self.adaptive_params = {
            'volume_spike_threshold': 1.2,  # 降低到1.2
            'min_signal_strength': 0.3,     # 降低到0.3
            'rsi_oversold': 30,              # 放宽到30
            'rsi_overbought': 75             # 放宽到75
        }

        self.params = {**self.base_params, **self.adaptive_params}

        self.exchange = ccxt.binance()
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.position = 0
        self.trades = []
        self.last_trade_time = None
        self.position_entry_time = None
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0

        logging.info("🤖 自适应优化版ETH交易机器人已启动")
        logging.info(f"💰 初始资金: ${self.balance:.2f}")
        logging.info(f"📊 自适应参数: 成交量阈值={self.params['volume_spike_threshold']}, 信号强度={self.params['min_signal_strength']}")

    def fetch_market_data(self, limit=100):
        """获取市场数据"""
        try:
            ohlcv = self.exchange.fetch_ohlcv('ETH/USDT', '5m', limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logging.error(f"获取数据失败: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df):
        """计算技术指标"""
        if len(df) < self.params['bb_period']:
            return df

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.params['rsi_period']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 移动平均线
        df['ma_short'] = df['close'].rolling(window=self.params['ma_short']).mean()
        df['ma_long'] = df['close'].rolling(window=self.params['ma_long']).mean()

        # 布林带
        df['bb_middle'] = df['close'].rolling(window=self.params['bb_period']).mean()
        bb_std = df['close'].rolling(window=self.params['bb_period']).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * self.params['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (bb_std * self.params['bb_std'])

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = tr.rolling(window=self.params['atr_period']).mean()

        # 成交量指标
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # 动量指标
        df['momentum'] = df['close'].pct_change(periods=5)

        return df

    def generate_signals(self, df):
        """生成交易信号"""
        if len(df) < self.params['bb_period']:
            return df

        df = df.copy()

        # RSI信号
        df['rsi_signal'] = 0
        df.loc[df['rsi'] < self.params['rsi_oversold'], 'rsi_signal'] = 1
        df.loc[df['rsi'] > self.params['rsi_overbought'], 'rsi_signal'] = -1

        # MA信号
        df['ma_signal'] = 0
        df.loc[df['ma_short'] > df['ma_long'], 'ma_signal'] = 1
        df.loc[df['ma_short'] < df['ma_long'], 'ma_signal'] = -1

        # 布林带信号
        df['bb_signal'] = 0
        df.loc[df['close'] < df['bb_lower'], 'bb_signal'] = 1
        df.loc[df['close'] > df['bb_upper'], 'bb_signal'] = -1

        # 成交量确认（降低要求）
        df['volume_confirmed'] = df['volume_ratio'] > self.params['volume_spike_threshold']

        # 综合信号强度
        signal_columns = ['rsi_signal', 'ma_signal', 'bb_signal']
        df['signal_strength'] = df[signal_columns].sum(axis=1)
        df['signal_strength'] = df['signal_strength'] / len(signal_columns)

        # 最终信号（动量条件放宽）
        df['final_signal'] = 0
        buy_condition = (
            (df['signal_strength'] >= self.params['min_signal_strength']) &
            (df['volume_confirmed']) &
            (df['momentum'] > -0.01)  # 允许小幅负动量
        )
        sell_condition = (
            (df['signal_strength'] <= -self.params['min_signal_strength']) &
            (df['volume_confirmed']) &
            (df['momentum'] < 0.01)  # 允许小幅正动量
        )

        df.loc[buy_condition, 'final_signal'] = 1
        df.loc[sell_condition, 'final_signal'] = -1

        return df

    def get_current_price(self):
        """获取当前价格"""
        try:
            ticker = self.exchange.fetch_ticker('ETH/USDT')
            return float(ticker['last'])
        except Exception as e:
            logging.error(f"获取价格失败: {e}")
            return None

    def calculate_dynamic_stops(self, entry_price, atr, direction):
        """计算动态止损止盈"""
        if not atr or atr == 0:
            if direction == 'long':
                stop_loss = entry_price * 0.98
                take_profit = entry_price * 1.04
            else:
                stop_loss = entry_price * 1.02
                take_profit = entry_price * 0.96
        else:
            stop_distance = atr * self.params['atr_multiplier']
            if direction == 'long':
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + (stop_distance * 2)
            else:
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - (stop_distance * 2)

        return stop_loss, take_profit

    def open_position(self, signal, current_price, df):
        """开仓"""
        if self.position != 0:
            return False

        # 时间过滤
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < self.params['min_time_between_trades']:
                return False

        atr = df.iloc[-1]['atr'] if 'atr' in df.columns else 0

        # 计算仓位大小
        risk_amount = self.balance * self.params['risk_per_trade']
        if atr > 0:
            stop_distance = atr * self.params['atr_multiplier']
            quantity = risk_amount / stop_distance
        else:
            quantity = risk_amount / (current_price * 0.02)

        quantity = min(quantity, self.balance * 0.1)

        try:
            if signal > 0:
                self.position = quantity
                self.entry_price = current_price
                self.stop_loss, self.take_profit = self.calculate_dynamic_stops(current_price, atr, 'long')
                self.position_entry_time = datetime.now()
                self.balance -= quantity

                logging.info(f"🟢 开多仓: {quantity:.4f} ETH @ ${current_price:.2f}")
                logging.info(f"🛑 止损: ${self.stop_loss:.2f}, 🎯 止盈: ${self.take_profit:.2f}")

            elif signal < 0:
                self.position = -quantity
                self.entry_price = current_price
                self.stop_loss, self.take_profit = self.calculate_dynamic_stops(current_price, atr, 'short')
                self.position_entry_time = datetime.now()
                self.balance -= quantity

                logging.info(f"🔴 开空仓: {quantity:.4f} ETH @ ${current_price:.2f}")
                logging.info(f"🛑 止损: ${self.stop_loss:.2f}, 🎯 止盈: ${self.take_profit:.2f}")

            self.last_trade_time = datetime.now()
            return True

        except Exception as e:
            logging.error(f"开仓失败: {e}")
            return False

    def manage_position(self, current_price):
        """管理持仓"""
        if self.position == 0:
            return False

        # 持仓时间检查
        if self.position_entry_time:
            holding_time = (datetime.now() - self.position_entry_time).total_seconds()
            if holding_time > self.params['max_holding_time']:
                return self.close_position(current_price, 'time_stop')

        # 止损止盈
        if self.position > 0:
            if current_price <= self.stop_loss or current_price >= self.take_profit:
                reason = 'stop_loss' if current_price <= self.stop_loss else 'take_profit'
                return self.close_position(current_price, reason)

        elif self.position < 0:
            if current_price >= self.stop_loss or current_price <= self.take_profit:
                reason = 'stop_loss' if current_price >= self.stop_loss else 'take_profit'
                return self.close_position(current_price, reason)

        return False

    def close_position(self, current_price, reason):
        """平仓"""
        if self.position == 0:
            return False

        try:
            if self.position > 0:
                pnl = (current_price - self.entry_price) * self.position
                self.balance += self.position
                trade_type = "多单"
            else:
                pnl = (self.entry_price - current_price) * abs(self.position)
                self.balance += abs(self.position)
                trade_type = "空单"

            trade = {
                'type': trade_type,
                'entry_time': self.position_entry_time,
                'exit_time': datetime.now(),
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'quantity': abs(self.position),
                'pnl': pnl,
                'balance': self.balance,
                'exit_reason': reason
            }
            self.trades.append(trade)

            reason_map = {
                'stop_loss': '止损',
                'take_profit': '止盈',
                'time_stop': '时间止损',
                'end_of_test': '测试结束'
            }
            reason_cn = reason_map.get(reason, reason)

            logging.info(f"✅ 平{trade_type}: {abs(self.position):.4f} ETH @ ${current_price:.2f}")
            logging.info(f"📊 盈亏: ${pnl:.2f} | 原因: {reason_cn} | 余额: ${self.balance:.2f}")

            self.position = 0
            self.position_entry_time = None
            self.entry_price = 0
            self.stop_loss = 0
            self.take_profit = 0

            return True

        except Exception as e:
            logging.error(f"平仓失败: {e}")
            return False

    def calculate_performance(self):
        """计算性能指标"""
        if not self.trades:
            return {}

        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        total_pnl = sum(t['pnl'] for t in self.trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        return_pct = (self.balance - self.initial_balance) / self.initial_balance

        profit_factor = abs(sum(t['pnl'] for t in winning_trades)) / abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf')

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'return_pct': return_pct,
            'profit_factor': profit_factor,
            'current_balance': self.balance
        }

    def run_test(self, duration_minutes=30):
        """运行测试"""
        logging.info(f"🚀 开始 {duration_minutes} 分钟自适应测试")
        logging.info(f"📅 开始时间: {datetime.now()}")

        start_time = datetime.now()
        test_end_time = start_time + timedelta(minutes=duration_minutes)

        while datetime.now() < test_end_time:
            try:
                # 获取数据
                df = self.fetch_market_data(limit=100)
                if df.empty:
                    time.sleep(30)
                    continue

                df = self.calculate_indicators(df)
                df = self.generate_signals(df)

                if len(df) < self.params['bb_period']:
                    time.sleep(30)
                    continue

                current_price = self.get_current_price()
                if not current_price:
                    time.sleep(30)
                    continue

                # 管理现有持仓
                if self.position != 0:
                    self.manage_position(current_price)

                # 检查新信号
                else:
                    latest_signal = df.iloc[-1]['final_signal']
                    if latest_signal != 0:
                        self.open_position(latest_signal, current_price, df)
                        logging.info(f"📈 信号触发: {latest_signal}, 强度: {df.iloc[-1]['signal_strength']:.2f}")

                # 状态显示
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                if int(elapsed) % 5 == 0:
                    performance = self.calculate_performance()
                    if performance:
                        logging.info(f"⏰ {elapsed:.0f}分钟 | 余额: ${self.balance:.2f} | 交易: {performance['total_trades']} | 胜率: {performance['win_rate']:.1%}")

                time.sleep(60)

            except KeyboardInterrupt:
                logging.info("🛑 用户中断测试")
                break
            except Exception as e:
                logging.error(f"❌ 测试出错: {e}")
                time.sleep(30)

        # 平仓
        if self.position != 0:
            current_price = self.get_current_price()
            if current_price:
                self.close_position(current_price, 'end_of_test')

        # 最终结果
        final_performance = self.calculate_performance()
        if final_performance:
            logging.info("="*50)
            logging.info("🏁 测试完成！")
            logging.info(f"💰 最终余额: ${final_performance['current_balance']:.2f}")
            logging.info(f"📊 总交易: {final_performance['total_trades']}")
            logging.info(f"🎯 胜率: {final_performance['win_rate']:.1%}")
            logging.info(f"💵 总盈亏: ${final_performance['total_pnl']:.2f}")
            logging.info(f"📈 收益率: {final_performance['return_pct']:.1%}")

        return final_performance

def main():
    try:
        trader = AdaptiveOptimizedScalper(initial_balance=10000)
        results = trader.run_test(duration_minutes=30)  # 30分钟测试

        if results:
            save_data = {
                'test_date': datetime.now().isoformat(),
                'strategy': 'adaptive_optimized',
                'parameters': trader.params,
                'performance': results
            }

            with open('adaptive_results.json', 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            logging.info("📁 结果已保存到 adaptive_results.json")

    except Exception as e:
        logging.error(f"主程序失败: {e}")

if __name__ == "__main__":
    main()