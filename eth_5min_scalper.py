#!/usr/bin/env python3
"""
ETH 5分钟高频剥头皮交易机器人
专为5分钟K线优化的高频交易系统
"""

import requests
import time
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import threading
import queue

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eth_5min_scalper.log'),
        logging.StreamHandler()
    ]
)

class ETH5MinScalper:
    def __init__(self):
        # 基础配置
        self.base_url = "https://api.binance.com/api/v3"
        self.fallback_url = "https://api.coingecko.com/api/v3"

        # 5分钟周期优化参数
        self.scalping_params = {
            # 更激进的止盈止损 (适应5分钟高频交易)
            'take_profit_pct': 0.003,      # 0.3% 止盈 (5分钟级别)
            'stop_loss_pct': 0.005,        # 0.5% 止损

            # 更敏感的技术指标
            'rsi_period': 7,               # 7周期RSI (适应5分钟)
            'rsi_oversold': 35,            # 更敏感的RSI阈值
            'rsi_overbought': 65,
            'ma_period_short': 12,         # 12周期短期MA (1小时)
            'ma_period_long': 48,          # 48周期长期MA (4小时)

            # 价格变动阈值
            'momentum_threshold': 0.001,   # 0.1% 动量阈值
            'volume_spike_threshold': 2.0, # 成交量激增阈值

            # 资金管理 (更积极)
            'position_size_ratio': 0.20,   # 20% 头寸比例
            'max_position_size': 500,       # 较小头寸，高频交易
            'risk_per_trade': 0.01,        # 单笔风险1%

            # 时间管理 (快速进出)
            'max_holding_time': 1800,      # 30分钟最大持仓
            'min_holding_time': 300,       # 5分钟最小持仓

            'initial_balance': 10000
        }

        # 实时数据存储
        self.price_data = []
        self.volume_data = []
        self.indicators = {}

        # 交易状态
        self.position = None
        self.balance = self.scalping_params['initial_balance']
        self.total_profit = 0
        self.trades_count = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # 信号系统
        self.signal_history = []
        self.last_signal_time = None
        self.signal_cooldown = 300  # 5分钟信号冷却

        # 性能监控
        self.start_time = datetime.now()
        self.daily_trades = []

        # 运行控制
        self.running = False
        self.check_interval = 60  # 1分钟检查间隔

    def get_current_price_binance(self) -> Optional[Dict]:
        """从Binance获取当前价格数据"""
        try:
            url = f"{self.base_url}/ticker/24hr"
            params = {'symbol': 'ETHUSDT'}

            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()

            data = response.json()

            return {
                'price': float(data['lastPrice']),
                'change_24h': float(data['priceChangePercent']),
                'volume_24h': float(data['volume']),
                'high_24h': float(data['highPrice']),
                'low_24h': float(data['lowPrice']),
                'bid': float(data['bidPrice']),
                'ask': float(data['askPrice']),
                'spread': float(data['askPrice']) - float(data['bidPrice']),
                'timestamp': datetime.now()
            }

        except Exception as e:
            logging.error(f"从Binance获取价格失败: {e}")
            return None

    def get_5min_klines(self, limit: int = 100) -> Optional[pd.DataFrame]:
        """获取5分钟K线数据"""
        try:
            url = f"{self.base_url}/klines"
            params = {
                'symbol': 'ETHUSDT',
                'interval': '5m',
                'limit': limit
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            df_data = []
            for kline in data:
                df_data.append({
                    'timestamp': pd.to_datetime(kline[0], unit='ms'),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })

            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            logging.error(f"获取5分钟K线失败: {e}")
            return None

    def calculate_5min_indicators(self, df: pd.DataFrame) -> Dict:
        """计算5分钟周期的技术指标"""
        if len(df) < self.scalping_params['ma_period_long']:
            return {}

        indicators = {}

        # 价格数据
        close_prices = df['close']
        high_prices = df['high']
        low_prices = df['low']
        volumes = df['volume']

        # 移动平均线
        indicators['ma_short'] = close_prices.rolling(window=self.scalping_params['ma_period_short']).mean()
        indicators['ma_long'] = close_prices.rolling(window=self.scalping_params['ma_period_long']).mean()

        # RSI
        def calculate_rsi(prices, period):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        indicators['rsi'] = calculate_rsi(close_prices, self.scalping_params['rsi_period'])

        # 布林带 (20周期)
        bb_period = 20
        bb_std = 2
        indicators['bb_middle'] = close_prices.rolling(window=bb_period).mean()
        bb_std_val = close_prices.rolling(window=bb_period).std()
        indicators['bb_upper'] = indicators['bb_middle'] + (bb_std_val * bb_std)
        indicators['bb_lower'] = indicators['bb_middle'] - (bb_std_val * bb_std)

        # 动量指标
        indicators['momentum_5min'] = close_prices.pct_change(1)  # 5分钟动量
        indicators['momentum_15min'] = close_prices.pct_change(3)  # 15分钟动量
        indicators['momentum_30min'] = close_prices.pct_change(6)  # 30分钟动量

        # 成交量指标
        indicators['volume_ma'] = volumes.rolling(window=20).mean()
        indicators['volume_ratio'] = volumes / indicators['volume_ma']

        # 价格波动率
        indicators['volatility'] = close_prices.rolling(window=10).std() / close_prices.rolling(window=10).mean()

        # 价格通道
        indicators['highest_20min'] = high_prices.rolling(window=4).max()  # 20分钟最高
        indicators['lowest_20min'] = low_prices.rolling(window=4).min()    # 20分钟最低

        # MACD (快参数适应5分钟)
        ema_12 = close_prices.ewm(span=12).mean()
        ema_26 = close_prices.ewm(span=26).mean()
        indicators['macd'] = ema_12 - ema_26
        indicators['macd_signal'] = indicators['macd'].ewm(span=6).mean()

        return indicators

    def generate_5min_signals(self, current_price: float, indicators: Dict, market_data: Dict) -> Dict:
        """生成5分钟周期交易信号"""
        if not indicators:
            return {'signal': 'hold', 'strength': 0, 'reason': '指标不足'}

        signals = []
        reasons = []

        latest_rsi = indicators['rsi'].iloc[-1] if not indicators['rsi'].empty else 50
        latest_ma_short = indicators['ma_short'].iloc[-1] if not indicators['ma_short'].empty else current_price
        latest_ma_long = indicators['ma_long'].iloc[-1] if not indicators['ma_long'].empty else current_price
        latest_volume_ratio = indicators['volume_ratio'].iloc[-1] if not indicators['volume_ratio'].empty else 1
        latest_volatility = indicators['volatility'].iloc[-1] if not indicators['volatility'].empty else 0.01

        # 买入信号条件
        buy_score = 0

        # RSI超卖
        if latest_rsi < self.scalping_params['rsi_oversold']:
            buy_score += 0.3
            signals.append('rsi_oversold')
            reasons.append(f'RSI超卖({latest_rsi:.1f})')

        # MA金叉
        if latest_ma_short > latest_ma_long:
            buy_score += 0.2
            signals.append('ma_golden_cross')
            reasons.append(f'MA金叉({latest_ma_short:.2f}>{latest_ma_long:.2f})')

        # 价格突破阻力
        if 'highest_20min' in indicators and not indicators['highest_20min'].empty:
            if current_price > indicators['highest_20min'].iloc[-1] * 1.001:
                buy_score += 0.25
                signals.append('price_breakout_up')
                reasons.append('价格突破20分钟高点')

        # 成交量激增配合价格上涨
        if latest_volume_ratio > self.scalping_params['volume_spike_threshold']:
            if 'momentum_5min' in indicators and not indicators['momentum_5min'].empty:
                if indicators['momentum_5min'].iloc[-1] > self.scalping_params['momentum_threshold']:
                    buy_score += 0.2
                    signals.append('volume_spike_up')
                    reasons.append(f'成交量激增{latest_volume_ratio:.1f}倍+价格上涨')

        # 布林带下轨支撑
        if 'bb_lower' in indicators and not indicators['bb_lower'].empty:
            if current_price <= indicators['bb_lower'].iloc[-1] * 1.002:
                buy_score += 0.15
                signals.append('bb_support')
                reasons.append('触及布林带下轨')

        # 卖出信号条件
        sell_score = 0

        # RSI超买
        if latest_rsi > self.scalping_params['rsi_overbought']:
            sell_score += 0.3
            reasons.append(f'RSI超买({latest_rsi:.1f})')

        # MA死叉
        if latest_ma_short < latest_ma_long:
            sell_score += 0.2
            reasons.append(f'MA死叉({latest_ma_short:.2f}<{latest_ma_long:.2f})')

        # 价格跌破支撑
        if 'lowest_20min' in indicators and not indicators['lowest_20min'].empty:
            if current_price < indicators['lowest_20min'].iloc[-1] * 0.999:
                sell_score += 0.25
                reasons.append('价格跌破20分钟低点')

        # MACD死叉
        if 'macd' in indicators and 'macd_signal' in indicators:
            if not indicators['macd'].empty and not indicators['macd_signal'].empty:
                if indicators['macd'].iloc[-1] < indicators['macd_signal'].iloc[-1]:
                    sell_score += 0.15
                    reasons.append('MACD死叉')

        # 确定最终信号
        signal_strength = abs(buy_score - sell_score)

        if buy_score > sell_score and signal_strength > 0.4:
            final_signal = 'strong_buy' if signal_strength > 0.7 else 'buy'
        elif sell_score > buy_score and signal_strength > 0.4:
            final_signal = 'strong_sell' if signal_strength > 0.7 else 'sell'
        else:
            final_signal = 'hold'
            reasons.append('信号强度不足')

        return {
            'signal': final_signal,
            'strength': signal_strength,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'reasons': reasons,
            'indicators': {
                'rsi': latest_rsi,
                'ma_short': latest_ma_short,
                'ma_long': latest_ma_long,
                'volume_ratio': latest_volume_ratio,
                'volatility': latest_volatility
            }
        }

    def calculate_position_size(self, signal_strength: float, current_price: float, spread: float) -> float:
        """计算5分钟高频交易的头寸大小"""
        base_size = self.scalping_params['max_position_size']

        # 根据信号强度调整
        strength_multiplier = min(signal_strength * 1.5, 1.0)

        # 根据价差调整 (价差大时减少头寸)
        spread_percentage = spread / current_price
        spread_penalty = max(0.5, 1 - spread_percentage * 100)

        # 根据波动率调整
        current_volatility = self.indicators.get('volatility', pd.Series([0.01])).iloc[-1] if self.indicators else 0.01
        volatility_adjustment = min(1.0, 0.5 / current_volatility) if current_volatility > 0 else 1.0

        # 计算最终头寸
        optimal_size = base_size * strength_multiplier * spread_penalty * volatility_adjustment

        # 余额限制
        balance_limit = self.balance * self.scalping_params['position_size_ratio']

        return min(optimal_size, balance_limit)

    def check_5min_position_exit(self, current_price: float, current_signal: Dict) -> Optional[str]:
        """检查5分钟持仓的平仓条件"""
        if self.position is None:
            return None

        position_type = self.position['type']
        entry_price = self.position['entry_price']
        entry_time = self.position['entry_time']
        holding_time = (datetime.now() - entry_time).total_seconds()

        # 计算当前盈亏
        if position_type == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # short
            pnl_pct = (entry_price - current_price) / entry_price

        exit_reason = None

        # 1. 止盈止损
        if position_type == 'long':
            if current_price <= self.position['stop_loss']:
                exit_reason = '止损'
            elif current_price >= self.position['take_profit']:
                exit_reason = '止盈'
        else:  # short
            if current_price >= self.position['stop_loss']:
                exit_reason = '止损'
            elif current_price <= self.position['take_profit']:
                exit_reason = '止盈'

        # 2. 反向信号 (快速响应)
        if not exit_reason:
            if (position_type == 'long' and current_signal['signal'] in ['sell', 'strong_sell']) or \
               (position_type == 'short' and current_signal['signal'] in ['buy', 'strong_buy']):
                if current_signal['strength'] > 0.6:  # 强信号立即平仓
                    exit_reason = f"强反向信号({current_signal['signal']})"

        # 3. 时间止损
        if not exit_reason:
            if holding_time > self.scalping_params['max_holding_time']:
                exit_reason = '时间止损(30分钟)'
            elif holding_time > self.scalping_params['min_holding_time'] and abs(pnl_pct) > 0.001:  # 5分钟后有微小盈利可平仓
                if abs(pnl_pct) > 0.002:  # 0.2%以上可平仓
                    exit_reason = '时间获利了结'

        # 4. 动态止损 (跟踪止损)
        if not exit_reason and holding_time > self.scalping_params['min_holding_time']:
            if position_type == 'long' and pnl_pct > 0.001:  # 多头盈利0.1%
                # 设置动态止损在入场价格
                if entry_price > self.position.get('dynamic_stop', 0):
                    self.position['dynamic_stop'] = entry_price
                    if current_price <= entry_price * 0.999:  # 回撤0.1%平仓
                        exit_reason = '动态止损'

            elif position_type == 'short' and pnl_pct > 0.001:  # 空头盈利0.1%
                if entry_price < self.position.get('dynamic_stop', float('inf')):
                    self.position['dynamic_stop'] = entry_price
                    if current_price >= entry_price * 1.001:  # 回撤0.1%平仓
                        exit_reason = '动态止损'

        return exit_reason

    def execute_5min_trade(self, signal: Dict, current_price: float, market_data: Dict) -> bool:
        """执行5分钟高频交易"""
        if self.position is not None:
            return False

        signal_type = signal['signal']
        if signal_type not in ['buy', 'strong_buy', 'sell', 'strong_sell']:
            return False

        # 信号冷却检查
        current_time = datetime.now()
        if (self.last_signal_time and
            (current_time - self.last_signal_time).total_seconds() < self.signal_cooldown):
            return False

        # 计算头寸大小
        position_size = self.calculate_position_size(
            signal['strength'],
            current_price,
            market_data.get('spread', 1.0)
        )

        if position_size <= 0:
            return False

        # 确定头寸类型
        position_type = 'long' if signal_type in ['buy', 'strong_buy'] else 'short'

        # 创建头寸
        self.position = {
            'type': position_type,
            'entry_price': current_price,
            'size': position_size,
            'entry_time': current_time,
            'signal_strength': signal['strength'],
            'signal_reasons': signal['reasons'],

            # 动态止盈止损
            'stop_loss': current_price * (1 - self.scalping_params['stop_loss_pct']) if position_type == 'long'
                      else current_price * (1 + self.scalping_params['stop_loss_pct']),
            'take_profit': current_price * (1 + self.scalping_params['take_profit_pct']) if position_type == 'long'
                        else current_price * (1 - self.scalping_params['take_profit_pct']),

            'dynamic_stop': None
        }

        self.last_signal_time = current_time

        logging.info(f"🚀 5分钟开仓: {signal_type}")
        logging.info(f"💰 头寸: ${position_size:.2f} @ ${current_price:.2f}")
        logging.info(f"📊 信号强度: {signal['strength']:.3f}")
        logging.info(f"📈 理由: {', '.join(signal['reasons'])}")
        logging.info(f"⛔ 止损: ${self.position['stop_loss']:.2f}")
        logging.info(f"🎯 止盈: ${self.position['take_profit']:.2f}")

        return True

    def close_5min_position(self, current_price: float, reason: str = ""):
        """平仓5分钟持仓"""
        if self.position is None:
            return

        entry_price = self.position['entry_price']
        position_type = self.position['type']
        position_size = self.position['size']
        entry_time = self.position['entry_time']
        holding_time = datetime.now() - entry_time

        # 计算盈亏
        if position_type == 'long':
            pnl = (current_price - entry_price) / entry_price * position_size
        else:  # short
            pnl = (entry_price - current_price) / entry_price * position_size

        # 更新账户
        self.balance += pnl
        self.total_profit += pnl
        self.trades_count += 1

        if pnl > 0:
            self.winning_trades += 1
            result_emoji = "✅"
        else:
            self.losing_trades += 1
            result_emoji = "❌"

        # 计算统计
        win_rate = self.winning_trades / self.trades_count if self.trades_count > 0 else 0
        pnl_percentage = pnl / position_size * 100

        # 记录交易
        trade_record = {
            'time': datetime.now(),
            'type': position_type,
            'entry_price': entry_price,
            'exit_price': current_price,
            'size': position_size,
            'pnl': pnl,
            'pnl_pct': pnl_percentage,
            'holding_time': holding_time.total_seconds(),
            'reason': reason
        }
        self.daily_trades.append(trade_record)

        logging.info(f"{result_emoji} 5分钟平仓: {position_type} ${position_size:.2f}")
        logging.info(f"📊 入场: ${entry_price:.2f} → 出场: ${current_price:.2f}")
        logging.info(f"💵 盈亏: ${pnl:+.2f} ({pnl_percentage:+.2f}%)")
        logging.info(f"⏱️ 持仓: {holding_time}")
        logging.info(f"🏆 当前余额: ${self.balance:.2f}")
        logging.info(f"📈 总盈亏: ${self.total_profit:+.2f}")
        logging.info(f"🎯 交易统计: {self.trades_count}笔, 胜率: {win_rate:.1%}")

        self.position = None

    def run_5min_scalping_loop(self):
        """运行5分钟剥头皮交易循环"""
        logging.info("🚀 启动ETH 5分钟高频剥头皮交易机器人")
        logging.info("⚡ 专为5分钟K线优化的超高频交易系统")
        logging.info(f"💰 初始资金: ${self.scalping_params['initial_balance']}")

        self.running = True
        cycle_count = 0

        while self.running:
            try:
                cycle_count += 1

                # 获取实时价格数据
                price_data = self.get_current_price_binance()
                if not price_data:
                    logging.warning("❌ 获取价格数据失败，等待下次尝试")
                    time.sleep(self.check_interval)
                    continue

                current_price = price_data['price']

                # 获取5分钟K线数据
                kline_data = self.get_5min_klines(100)
                if kline_data is None or len(kline_data) < 50:
                    logging.warning("❌ K线数据不足")
                    time.sleep(self.check_interval)
                    continue

                # 计算技术指标
                self.indicators = self.calculate_5min_indicators(kline_data)

                # 生成交易信号
                market_data = {
                    'spread': price_data.get('spread', 1.0),
                    'bid': price_data.get('bid', current_price),
                    'ask': price_data.get('ask', current_price),
                    'volume_24h': price_data.get('volume_24h', 0)
                }

                signal_result = self.generate_5min_signals(current_price, self.indicators, market_data)

                # 检查持仓状态
                if self.position:
                    exit_reason = self.check_5min_position_exit(current_price, signal_result)
                    if exit_reason:
                        self.close_5min_position(current_price, exit_reason)
                    else:
                        # 显示持仓状态
                        entry_price = self.position['entry_price']
                        if self.position['type'] == 'long':
                            unrealized_pnl = (current_price - entry_price) / entry_price
                        else:
                            unrealized_pnl = (entry_price - current_price) / entry_price

                        holding_time = datetime.now() - self.position['entry_time']

                        logging.info(f"📊 持仓: {self.position['type']} | "
                                   f"盈亏: {unrealized_pnl*100:+.2f}% | "
                                   f"时间: {holding_time.total_seconds()/60:.1f}min | "
                                   f"价格: ${current_price:.2f}")

                # 如果没有持仓且有信号，执行交易
                if not self.position and signal_result['signal'] in ['buy', 'strong_buy', 'sell', 'strong_sell']:
                    success = self.execute_5min_trade(signal_result, current_price, market_data)
                    if not success:
                        logging.debug(f"交易执行失败: {signal_result['reasons']}")

                # 显示当前状态
                if cycle_count % 10 == 0:  # 每10个周期显示一次
                    latest_rsi = self.indicators.get('rsi', pd.Series([50])).iloc[-1] if self.indicators else 50
                    latest_ma_short = self.indicators.get('ma_short', pd.Series([current_price])).iloc[-1] if self.indicators else current_price

                    logging.info(f"📊 信号: {signal_result['signal']} "
                               f"(强度: {signal_result['strength']:.3f})")
                    logging.info(f"💹 价格: ${current_price:.2f} "
                               f"RSI: {latest_rsi:.1f} "
                               f"MA短: ${latest_ma_short:.2f} "
                               f"价差: ${market_data.get('spread', 0):.2f}")

                # 每30个周期显示详细统计
                if cycle_count % 30 == 0:
                    self.print_5min_performance_summary()

                # 5分钟周期的主要检查间隔
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logging.info("🛑 收到停止信号，正在平仓...")
                self.running = False
                if self.position:
                    price_data = self.get_current_price_binance()
                    if price_data:
                        self.close_5min_position(price_data['price'], "手动停止")
                self.print_5min_performance_summary()
                break

            except Exception as e:
                logging.error(f"❌ 交易循环出错: {e}")
                time.sleep(self.check_interval)

    def print_5min_performance_summary(self):
        """打印5分钟高频交易性能摘要"""
        if self.trades_count == 0:
            logging.info("📊 还没有执行任何交易")
            return

        win_rate = self.winning_trades / self.trades_count
        total_return = (self.balance - self.scalping_params['initial_balance']) / self.scalping_params['initial_balance']
        avg_trade = self.total_profit / self.trades_count

        # 计算今日交易统计
        today_trades = len(self.daily_trades)
        today_profit = sum(t['pnl'] for t in self.daily_trades)
        today_win_rate = sum(1 for t in self.daily_trades if t['pnl'] > 0) / today_trades if today_trades > 0 else 0
        avg_holding_time = np.mean([t['holding_time'] for t in self.daily_trades]) if self.daily_trades else 0

        runtime = datetime.now() - self.start_time
        trades_per_hour = self.trades_count / max(runtime.total_seconds() / 3600, 1)

        logging.info("="*70)
        logging.info("🚀 ETH 5分钟高频交易性能摘要")
        logging.info("="*70)
        logging.info(f"💰 当前余额: ${self.balance:.2f}")
        logging.info(f"📈 总收益率: {total_return:+.2%}")
        logging.info(f"💵 总盈亏: ${self.total_profit:+.2f}")
        logging.info(f"🔢 总交易次数: {self.trades_count}")
        logging.info(f"✅ 盈利交易: {self.winning_trades} | ❌ 亏损交易: {self.losing_trades}")
        logging.info(f"🎯 总胜率: {win_rate:.1%}")
        logging.info(f"💹 平均每笔: ${avg_trade:+.2f}")
        logging.info(f"⚡ 交易频率: {trades_per_hour:.1f}笔/小时")
        logging.info(f"⏱️ 运行时间: {runtime}")

        if self.daily_trades:
            logging.info("-" * 70)
            logging.info("📊 今日交易详情:")
            logging.info(f"🔢 今日交易: {today_trades}笔")
            logging.info(f"💵 今日盈亏: ${today_profit:+.2f}")
            logging.info(f"🎯 今日胜率: {today_win_rate:.1%}")
            logging.info(f"⏱️ 平均持仓: {avg_holding_time/60:.1f}分钟")

        logging.info("="*70)

def main():
    """主函数"""
    print("🚀 ETH 5分钟高频剥头皮交易机器人")
    print("="*60)
    print("⚡ 专为5分钟K线优化的超高频交易系统")
    print("🎯 特点: 快速进出、精确止损、动态调整")
    print("⚠️  警告: 高风险高频交易，仅供学习研究")
    print("="*60)

    trader = ETH5MinScalper()

    try:
        trader.run_5min_scalping_loop()
    except Exception as e:
        logging.error(f"系统错误: {e}")
    finally:
        logging.info("🛑 交易机器人已停止")

if __name__ == "__main__":
    main()