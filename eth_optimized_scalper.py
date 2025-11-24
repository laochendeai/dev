#!/usr/bin/env python3
"""
ETH优化剥头皮交易机器人
基于回测最佳参数的实时交易版本
"""

import requests
import time
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eth_optimized_scalper.log'),
        logging.StreamHandler()
    ]
)

class ETHOptimizedScalper:
    def __init__(self):
        # 从回测获得的最佳参数
        self.best_params = {
            'take_profit_pct': 0.008,      # 0.8% 止盈
            'stop_loss_pct': 0.012,        # 1.2% 止损
            'rsi_oversold': 45,            # RSI超卖阈值
            'rsi_overbought': 55,          # RSI超买阈值
            'price_drop_threshold': 0.015, # 价格下跌阈值
            'price_rise_threshold': 0.015, # 价格上涨阈值
            'initial_balance': 10000,
            'max_position_size': 1200,
            'position_size_ratio': 0.12,
            'max_holding_time': 64800      # 18小时最大持仓时间
        }

        self.base_url = "https://api.coingecko.com/api/v3"
        self.price_history = []
        self.position = None
        self.balance = self.best_params['initial_balance']
        self.total_profit = 0
        self.trades_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.check_interval = 30  # 30秒检查间隔

    def get_current_price(self) -> Optional[float]:
        """获取ETH当前价格"""
        try:
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': 'ethereum',
                'vs_currencies': 'usd',
                'include_24hr_change': 'true'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            price = data['ethereum']['usd']
            price_change = data['ethereum'].get('usd_24h_change', 0)

            # 添加到历史记录
            self.price_history.append({
                'price': price,
                'change_24h': price_change,
                'timestamp': datetime.now()
            })

            # 保持历史记录在合理范围内
            if len(self.price_history) > 100:
                self.price_history.pop(0)

            logging.info(f"当前ETH价格: ${price:.2f} (24h变化: {price_change:.2f}%)")
            return price

        except Exception as e:
            logging.error(f"获取价格失败: {e}")
            return None

    def get_historical_prices(self, hours: int = 48) -> List[float]:
        """获取历史价格数据用于技术指标计算"""
        try:
            url = f"{self.base_url}/coins/ethereum/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': max(2, hours // 24 + 1),
            }

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()
            prices = [price[1] for price in data['prices']]

            return prices[-hours:] if len(prices) > hours else prices

        except Exception as e:
            logging.error(f"获取历史数据失败: {e}")
            time.sleep(1)  # 添加延迟避免频繁请求
            return []

    def calculate_indicators(self, prices: List[float]) -> dict:
        """计算技术指标"""
        if len(prices) < 20:
            return {}

        # 移动平均线
        ma_5 = np.mean(prices[-5:])
        ma_10 = np.mean(prices[-10:])
        ma_20 = np.mean(prices[-20:])

        # RSI
        def calculate_rsi(price_list, period=14):
            if len(price_list) < period + 1:
                return 50

            deltas = np.diff(price_list)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])

            if avg_loss == 0:
                return 100

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        rsi_14 = calculate_rsi(prices)
        rsi_7 = calculate_rsi(prices, 7)

        # 价格变动
        price_change_3h = (prices[-1] - prices[-3]) / prices[-3] if len(prices) > 3 else 0

        # 波动率
        volatility = np.std(prices[-10:]) / np.mean(prices[-10:])

        return {
            'ma_5': ma_5,
            'ma_10': ma_10,
            'ma_20': ma_20,
            'rsi_14': rsi_14,
            'rsi_7': rsi_7,
            'price_change_3h': price_change_3h,
            'volatility': volatility
        }

    def generate_optimized_signals(self, current_price: float, indicators: dict) -> str:
        """生成优化交易信号"""
        if not indicators:
            return 'hold'

        signals = []

        # RSI超卖买入信号
        if indicators['rsi_14'] < self.best_params['rsi_oversold']:
            signals.append('rsi_oversold_buy')

        # RSI 7周期超卖
        if indicators['rsi_7'] < 40:
            signals.append('rsi7_oversold_buy')

        # 价格大幅下跌买入
        if indicators['price_change_3h'] < -self.best_params['price_drop_threshold']:
            signals.append('price_drop_buy')

        # 移动平均线支撑买入
        if (current_price < indicators['ma_10'] and
            indicators['ma_5'] > indicators['ma_10']):
            signals.append('ma_support_buy')

        # RSI超买卖出信号
        if indicators['rsi_14'] > self.best_params['rsi_overbought']:
            signals.append('rsi_overbought_sell')

        # RSI 7周期超买
        if indicators['rsi_7'] > 60:
            signals.append('rsi7_overbought_sell')

        # 价格大幅上涨卖出
        if indicators['price_change_3h'] > self.best_params['price_rise_threshold']:
            signals.append('price_rise_sell')

        # 移动平均线阻力卖出
        if (current_price > indicators['ma_10'] and
            indicators['ma_5'] < indicators['ma_10']):
            signals.append('ma_resistance_sell')

        # 综合决策
        buy_signals = [s for s in signals if s.endswith('_buy')]
        sell_signals = [s for s in signals if s.endswith('_sell')]

        if buy_signals and not sell_signals:
            return 'buy'
        elif sell_signals and not buy_signals:
            return 'sell'
        else:
            return 'hold'

    def calculate_position_size(self, signal: str, current_price: float) -> float:
        """计算头寸大小"""
        base_size = self.best_params['max_position_size'] * 0.8  # 使用80%的最大头寸

        # 根据余额调整
        balance_based_size = self.balance * self.best_params['position_size_ratio']

        return min(base_size, balance_based_size)

    def enter_position(self, signal: str, current_price: float, indicators: dict) -> bool:
        """建立头寸"""
        if self.position is not None:
            logging.warning("已有持仓，跳过建仓")
            return False

        position_size = self.calculate_position_size(signal, current_price)
        if position_size <= 0:
            return False

        position_type = 'long' if signal == 'buy' else 'short'

        self.position = {
            'type': position_type,
            'entry_price': current_price,
            'size': position_size,
            'entry_time': datetime.now(),
            'stop_loss': current_price * (1 - self.best_params['stop_loss_pct']) if position_type == 'long'
                      else current_price * (1 + self.best_params['stop_loss_pct']),
            'take_profit': current_price * (1 + self.best_params['take_profit_pct']) if position_type == 'long'
                        else current_price * (1 - self.best_params['take_profit_pct']),
            'indicators': indicators
        }

        logging.info(f"建立{position_type}头寸: ${position_size:.2f} @ ${current_price:.2f}")
        logging.info(f"RSI: {indicators.get('rsi_14', 0):.1f}, MA5: {indicators.get('ma_5', 0):.2f}, MA10: {indicators.get('ma_10', 0):.2f}")
        logging.info(f"止损: ${self.position['stop_loss']:.2f}, 止盈: ${self.position['take_profit']:.2f}")

        return True

    def check_position(self, current_price: float, signal: str) -> Optional[dict]:
        """检查持仓状态"""
        if self.position is None:
            return None

        entry_price = self.position['entry_price']
        position_type = self.position['type']
        holding_time = (datetime.now() - self.position['entry_time']).total_seconds()

        # 计算未实现盈亏
        if position_type == 'long':
            unrealized_pnl = (current_price - entry_price) / entry_price
            should_close = (
                current_price <= self.position['stop_loss'] or
                current_price >= self.position['take_profit'] or
                signal == 'sell' or
                holding_time > self.best_params['max_holding_time']
            )
        else:  # short
            unrealized_pnl = (entry_price - current_price) / entry_price
            should_close = (
                current_price >= self.position['stop_loss'] or
                current_price <= self.position['take_profit'] or
                signal == 'buy' or
                holding_time > self.best_params['max_holding_time']
            )

        # 确定平仓原因
        close_reason = "未知原因"
        if should_close:
            if position_type == 'long':
                if current_price <= self.position['stop_loss']:
                    close_reason = "止损"
                elif current_price >= self.position['take_profit']:
                    close_reason = "止盈"
                elif signal == 'sell':
                    close_reason = "反向信号"
                elif holding_time > self.best_params['max_holding_time']:
                    close_reason = "时间止损"
            else:  # short
                if current_price >= self.position['stop_loss']:
                    close_reason = "止损"
                elif current_price <= self.position['take_profit']:
                    close_reason = "止盈"
                elif signal == 'buy':
                    close_reason = "反向信号"
                elif holding_time > self.best_params['max_holding_time']:
                    close_reason = "时间止损"

        return {
            'should_close': should_close,
            'close_reason': close_reason,
            'unrealized_pnl': unrealized_pnl,
            'holding_hours': holding_time / 3600
        }

    def close_position(self, current_price: float, reason: str = ""):
        """平仓"""
        if self.position is None:
            return

        entry_price = self.position['entry_price']
        position_type = self.position['type']
        position_size = self.position['size']
        holding_time = datetime.now() - self.position['entry_time']

        # 计算盈亏
        if position_type == 'long':
            pnl = (current_price - entry_price) / entry_price * position_size
        else:  # short
            pnl = (entry_price - current_price) / entry_price * position_size

        self.balance += pnl
        self.total_profit += pnl
        self.trades_count += 1

        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # 计算胜率
        win_rate = self.winning_trades / self.trades_count if self.trades_count > 0 else 0

        logging.info(f"平仓({reason}): {position_type} ${position_size:.2f}")
        logging.info(f"入场: ${entry_price:.2f}, 出场: ${current_price:.2f}")
        logging.info(f"盈亏: ${pnl:.2f} ({pnl/position_size*100:.2f}%)")
        logging.info(f"持仓时间: {holding_time}")
        logging.info(f"当前余额: ${self.balance:.2f}, 总盈亏: ${self.total_profit:.2f}")
        logging.info(f"交易统计: {self.trades_count}笔, 胜率: {win_rate:.1%}")

        self.position = None

    def print_performance_summary(self):
        """打印性能摘要"""
        if self.trades_count == 0:
            logging.info("还没有执行任何交易")
            return

        win_rate = self.winning_trades / self.trades_count
        total_return = (self.balance - self.best_params['initial_balance']) / self.best_params['initial_balance']
        avg_trade = self.total_profit / self.trades_count

        logging.info("=" * 50)
        logging.info("交易性能摘要")
        logging.info(f"总交易次数: {self.trades_count}")
        logging.info(f"盈利交易: {self.winning_trades}")
        logging.info(f"亏损交易: {self.losing_trades}")
        logging.info(f"胜率: {win_rate:.2%}")
        logging.info(f"总收益率: {total_return:.2%}")
        logging.info(f"平均每笔交易: ${avg_trade:.2f}")
        logging.info(f"当前余额: ${self.balance:.2f}")
        logging.info(f"总盈亏: ${self.total_profit:.2f}")
        logging.info("=" * 50)

    def run_optimized_trading(self):
        """运行优化交易循环"""
        logging.info("启动ETH优化剥头皮交易机器人...")
        logging.info(f"使用最佳参数: {self.best_params}")

        cycle_count = 0

        while True:
            try:
                cycle_count += 1
                current_price = self.get_current_price()

                if not current_price:
                    time.sleep(self.check_interval)
                    continue

                # 获取历史数据并计算指标
                historical_prices = self.get_historical_prices(48)
                if len(historical_prices) < 20:
                    logging.warning("历史数据不足，等待下次检查")
                    time.sleep(self.check_interval)
                    continue

                indicators = self.calculate_indicators(historical_prices)
                signal = self.generate_optimized_signals(current_price, indicators)

                # 检查现有持仓
                if self.position:
                    position_check = self.check_position(current_price, signal)
                    if position_check['should_close']:
                        self.close_position(current_price, position_check['close_reason'])
                    else:
                        # 显示持仓状态
                        pnl_percentage = position_check['unrealized_pnl'] * 100
                        logging.info(f"持仓状态: {self.position['type']} | "
                                   f"未实现盈亏: {pnl_percentage:+.3f}% | "
                                   f"持仓时间: {position_check['holding_hours']:.1f}小时")

                # 如果没有持仓且有强烈信号，考虑建仓
                if not self.position and signal in ['buy', 'sell']:
                    # 只有在指标支持的情况下才建仓
                    confidence_indicators = []
                    if signal == 'buy' and indicators['rsi_14'] < 45:
                        confidence_indicators.append('RSI支持')
                    if signal == 'sell' and indicators['rsi_14'] > 55:
                        confidence_indicators.append('RSI支持')

                    if confidence_indicators:
                        logging.info(f"交易信号: {signal} ({', '.join(confidence_indicators)})")
                        self.enter_position(signal, current_price, indicators)

                # 每10个循环显示一次详细状态
                if cycle_count % 10 == 0:
                    self.print_performance_summary()

                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logging.info("收到停止信号，正在平仓...")
                if self.position:
                    current_price = self.get_current_price()
                    if current_price:
                        self.close_position(current_price, "手动停止")
                self.print_performance_summary()
                break

            except Exception as e:
                logging.error(f"交易循环出错: {e}")
                time.sleep(self.check_interval)

def main():
    """主函数"""
    print("ETH 优化剥头皮交易机器人")
    print("=" * 40)
    print("基于历史数据回测的最佳参数")
    print("预期胜率: 64.42%")
    print("预期收益率: 2.65%")
    print("警告: 这仅用于教育和演示目的")
    print("实际交易存在风险，请谨慎使用")
    print("=" * 40)

    trader = ETHOptimizedScalper()
    trader.run_optimized_trading()

if __name__ == "__main__":
    main()