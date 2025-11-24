#!/usr/bin/env python3
"""
超高胜率剥头皮交易系统
目标胜率：80%+
基于订单簿分析 + 机器学习 + 市场微观结构
"""

import numpy as np
import pandas as pd
import ccxt
import time
import threading
import json
from datetime import datetime, timedelta
from collections import deque
import logging
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultra_high_winrate_scalper.log'),
        logging.StreamHandler()
    ]
)

class UltraHighWinrateScalper:
    """超高胜率剥头皮交易系统"""

    def __init__(self, initial_balance=10000):
        # 核心参数（基于研究的最佳实践）
        self.params = {
            # 订单簿分析参数
            'lob_levels': 10,                    # 订单簿深度层级
            'imbalance_threshold': 0.3,          # 订单不平衡阈值
            'spread_threshold': 0.001,           # 价差阈值 (0.1%)
            'liquidity_ratio_threshold': 0.6,    # 流动性比率阈值

            # 做市商参数
            'base_spread': 0.0005,               # 基础价差 0.05%
            'skew_adjustment': 0.002,            # 存货倾斜调整
            'inventory_limit': 0.3,              # 存货限制 30%
            'target_profit': 0.001,              # 目标利润 0.1%
            'max_loss': 0.0005,                  # 最大损失 0.05%

            # 风险控制参数
            'max_position_size': 0.05,           # 最大仓位 5%
            'max_daily_trades': 50,              # 最大日交易次数
            'heat_factor': 0.1,                  # 热度因子
            'latency_threshold': 0.1,            # 延迟阈值 100ms

            # 时间控制
            'holding_period_max': 300,           # 最大持仓时间 5分钟
            'cooldown_period': 10,               # 冷却时间 10秒

            # 信号权重
            'lob_weight': 0.4,                   # 订单簿权重
            'momentum_weight': 0.3,              # 动量权重
            'volatility_weight': 0.2,            # 波动率权重
            'volume_weight': 0.1                 # 成交量权重
        }

        # 初始化交易所连接
        self.exchange = ccxt.binance({
            'options': {
                'defaultType': 'future',
            }
        })

        # 账户状态
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.inventory = 0  # 当前持仓 (正数多头，负数空头)
        self.daily_trades = 0
        self.last_trade_time = None

        # 数据存储
        self.order_book_buffer = deque(maxlen=1000)
        self.trade_buffer = deque(maxlen=1000)
        self.price_history = deque(maxlen=1000)

        # 性能统计
        self.trades = []
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0
        self.heat_counter = 0

        logging.info("🚀 超高胜率剥头皮交易系统已启动")
        logging.info(f"💰 初始资金: ${self.balance:.2f}")
        logging.info(f"🎯 目标胜率: 80%+")

    def get_order_book(self):
        """获取实时订单簿数据"""
        try:
            orderbook = self.exchange.fetch_order_book('ETH/USDT', limit=50)

            # 处理买单
            bids = orderbook['bids'][:self.params['lob_levels']]
            bid_prices = [float(bid[0]) for bid in bids]
            bid_volumes = [float(bid[1]) for bid in bids]

            # 处理卖单
            asks = orderbook['asks'][:self.params['lob_levels']]
            ask_prices = [float(ask[0]) for ask in asks]
            ask_volumes = [float(ask[1]) for ask in asks]

            return {
                'timestamp': datetime.now(),
                'bid_prices': bid_prices,
                'bid_volumes': bid_volumes,
                'ask_prices': ask_prices,
                'ask_volumes': ask_volumes,
                'spread': ask_prices[0] - bid_prices[0],
                'mid_price': (ask_prices[0] + bid_prices[0]) / 2
            }
        except Exception as e:
            logging.error(f"获取订单簿失败: {e}")
            return None

    def calculate_lob_imbalance(self, lob_data):
        """计算订单簿不平衡性"""
        try:
            # 计算前N个层级的买卖量不平衡
            bid_volume = sum(lob_data['bid_volumes'][:5])
            ask_volume = sum(lob_data['ask_volumes'][:5])

            if (bid_volume + ask_volume) == 0:
                return 0

            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            return imbalance
        except:
            return 0

    def calculate_liquidity_ratio(self, lob_data):
        """计算流动性比率"""
        try:
            # 计算订单簿深度的不对称性
            total_bid_volume = sum(lob_data['bid_volumes'])
            total_ask_volume = sum(lob_data['ask_volumes'])

            if total_ask_volume == 0:
                return float('inf')

            ratio = total_bid_volume / total_ask_volume
            return min(ratio, 5)  # 限制最大值
        except:
            return 1

    def calculate_volume_weighted_price(self, lob_data):
        """计算成交量加权价格"""
        try:
            # VWAP计算
            total_volume = 0
            weighted_sum = 0

            # 买单VWAP
            for price, volume in zip(lob_data['bid_prices'], lob_data['bid_volumes']):
                weighted_sum += price * volume
                total_volume += volume

            bid_vwap = weighted_sum / total_volume if total_volume > 0 else lob_data['bid_prices'][0]

            # 卖单VWAP
            weighted_sum = 0
            total_volume = 0

            for price, volume in zip(lob_data['ask_prices'], lob_data['ask_volumes']):
                weighted_sum += price * volume
                total_volume += volume

            ask_vwap = weighted_sum / total_volume if total_volume > 0 else lob_data['ask_prices'][0]

            return (bid_vwap + ask_vwap) / 2
        except:
            return lob_data['mid_price']

    def calculate_order_flow(self):
        """计算订单流不平衡"""
        if len(self.trade_buffer) < 10:
            return 0

        # 分析最近的交易方向
        recent_trades = list(self.trade_buffer)[-10:]
        buy_volume = sum(trade['volume'] for trade in recent_trades if trade['side'] == 'buy')
        sell_volume = sum(trade['volume'] for trade in recent_trades if trade['side'] == 'sell')

        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0

        return (buy_volume - sell_volume) / total_volume

    def calculate_price_momentum(self):
        """计算价格动量"""
        if len(self.price_history) < 20:
            return 0

        prices = list(self.price_history)
        # 计算短期和长期移动平均线
        short_ma = np.mean(prices[-5:])
        long_ma = np.mean(prices[-20:])

        if long_ma == 0:
            return 0

        return (short_ma - long_ma) / long_ma

    def calculate_realized_volatility(self):
        """计算已实现波动率"""
        if len(self.price_history) < 20:
            return 0

        prices = list(self.price_history)
        returns = np.diff(prices) / prices[:-1]

        return np.std(returns) * np.sqrt(252)  # 年化波动率

    def generate_trading_signal(self, lob_data):
        """生成交易信号（基于多个因子）"""
        try:
            # 1. 订单簿不平衡信号
            lob_imbalance = self.calculate_lob_imbalance(lob_data)
            lob_signal = 1 if lob_imbalance > self.params['imbalance_threshold'] else (-1 if lob_imbalance < -self.params['imbalance_threshold'] else 0)

            # 2. 流动性比率信号
            liquidity_ratio = self.calculate_liquidity_ratio(lob_data)
            liquidity_signal = 1 if liquidity_ratio > self.params['liquidity_ratio_threshold'] else (-1 if liquidity_ratio < 1/self.params['liquidity_ratio_threshold'] else 0)

            # 3. 订单流信号
            order_flow = self.calculate_order_flow()
            flow_signal = 1 if order_flow > 0.2 else (-1 if order_flow < -0.2 else 0)

            # 4. 价格动量信号
            momentum = self.calculate_price_momentum()
            momentum_signal = 1 if momentum > 0.001 else (-1 if momentum < -0.001 else 0)

            # 5. 波动率过滤
            volatility = self.calculate_realized_volatility()
            volatility_ok = volatility > 0.01 and volatility < 0.5  # 避免过高波动

            # 综合信号（加权平均）
            if not volatility_ok:
                return 0, 0

            weighted_signal = (
                lob_signal * self.params['lob_weight'] +
                liquidity_signal * 0.2 +  # 流动性权重
                flow_signal * 0.1 +       # 订单流权重
                momentum_signal * self.params['momentum_weight']
            )

            # 计算信号强度
            signal_strength = abs(weighted_signal)

            # 信号确认（需要多个因子同向）
            confirmations = sum([
                lob_signal != 0,
                liquidity_signal != 0,
                flow_signal != 0,
                momentum_signal != 0
            ])

            # 至少2个因子确认且信号强度足够
            if confirmations >= 2 and signal_strength >= 0.3:
                final_signal = 1 if weighted_signal > 0 else -1
                return final_signal, signal_strength

            return 0, signal_strength

        except Exception as e:
            logging.error(f"信号生成失败: {e}")
            return 0, 0

    def calculate_optimal_quotes(self, lob_data):
        """计算最优报价（做市商策略）"""
        try:
            mid_price = lob_data['mid_price']
            spread = lob_data['spread']

            # 基础价差调整
            base_spread = self.params['base_spread'] * mid_price

            # 存货倾斜调整
            inventory_skew = self.inventory * self.params['skew_adjustment'] * mid_price

            # 波动率调整
            volatility = self.calculate_realized_volatility()
            volatility_adjustment = volatility * 0.1 * mid_price

            # 最终价差
            final_spread = max(base_spread, spread) + abs(inventory_skew) + volatility_adjustment

            # 最优买卖价
            optimal_bid = mid_price - final_spread / 2 + inventory_skew
            optimal_ask = mid_price + final_spread / 2 + inventory_skew

            return optimal_bid, optimal_ask

        except Exception as e:
            logging.error(f"最优报价计算失败: {e}")
            return lob_data['bid_prices'][0], lob_data['ask_prices'][0]

    def should_execute_trade(self, signal, signal_strength, lob_data):
        """判断是否应该执行交易"""
        try:
            # 1. 检查交易限制
            if self.daily_trades >= self.params['max_daily_trades']:
                return False

            # 2. 检查冷却时间
            if self.last_trade_time:
                time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
                if time_since_last < self.params['cooldown_period']:
                    return False

            # 3. 检查仓位限制
            position_ratio = abs(self.inventory) / self.balance
            if position_ratio >= self.params['inventory_limit']:
                return False

            # 4. 检查价差（确保有利可图）
            spread_ratio = lob_data['spread'] / lob_data['mid_price']
            if spread_ratio > self.params['spread_threshold'] * 5:  # 价差过大
                return False

            # 5. 检查信号强度
            if signal_strength < 0.3:  # 信号强度不够
                return False

            return True

        except Exception as e:
            logging.error(f"交易执行判断失败: {e}")
            return False

    def execute_trade(self, signal, lob_data):
        """执行交易"""
        try:
            current_price = lob_data['mid_price']

            # 计算仓位大小
            position_size = self.balance * self.params['max_position_size']

            # 应用热度因子调整
            if self.heat_counter > 0:
                position_size *= (1 - self.heat_counter * self.params['heat_factor'])

            quantity = position_size / current_price

            # 设置止损止盈
            if signal > 0:  # 买入
                stop_loss = current_price * (1 - self.params['max_loss'])
                take_profit = current_price * (1 + self.params['target_profit'])
                self.inventory += quantity
                trade_type = "买入"
            else:  # 卖出
                stop_loss = current_price * (1 + self.params['max_loss'])
                take_profit = current_price * (1 - self.params['target_profit'])
                self.inventory -= quantity
                trade_type = "卖出"

            # 记录交易
            trade = {
                'timestamp': datetime.now(),
                'type': trade_type,
                'signal': signal,
                'quantity': quantity,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'inventory': self.inventory
            }

            self.trades.append(trade)
            self.daily_trades += 1
            self.last_trade_time = datetime.now()

            logging.info(f"🟢 {trade_type}信号执行: {quantity:.6f} ETH @ ${current_price:.2f}")
            logging.info(f"🛑 止损: ${stop_loss:.2f}, 🎯 止盈: ${take_profit:.2f}")
            logging.info(f"📊 当前持仓: {self.inventory:.6f} ETH")

            return True

        except Exception as e:
            logging.error(f"交易执行失败: {e}")
            return False

    def monitor_positions(self, lob_data):
        """监控现有持仓"""
        if abs(self.inventory) < 0.001:  # 基本无持仓
            return

        current_price = lob_data['mid_price']

        # 检查最近的交易
        recent_trades = [t for t in self.trades if (datetime.now() - t['timestamp']).total_seconds() < 300]

        for trade in recent_trades:
            if trade.get('closed', False):
                continue

            holding_time = (datetime.now() - trade['timestamp']).total_seconds()

            # 止损检查
            if trade['signal'] > 0:  # 多头持仓
                if current_price <= trade['stop_loss'] or current_price >= trade['take_profit'] or holding_time > self.params['holding_period_max']:

                    # 平仓
                    pnl = (current_price - trade['entry_price']) * trade['quantity']

                    if current_price <= trade['stop_loss']:
                        reason = "止损"
                        self.losses += 1
                        self.heat_counter += 1
                    elif current_price >= trade['take_profit']:
                        reason = "止盈"
                        self.wins += 1
                        self.heat_counter = max(0, self.heat_counter - 1)
                    else:
                        reason = "时间止损"
                        self.heat_counter += 0.5

                    self.inventory -= trade['quantity']
                    self.total_pnl += pnl

                    logging.info(f"🔴 平仓 {trade['quantity']:.6f} ETH @ ${current_price:.2f} ({reason})")
                    logging.info(f"💰 盈亏: ${pnl:.2f}, 总盈亏: ${self.total_pnl:.2f}")

                    trade['closed'] = True
                    trade['exit_price'] = current_price
                    trade['exit_time'] = datetime.now()
                    trade['pnl'] = pnl
                    trade['exit_reason'] = reason

            elif trade['signal'] < 0:  # 空头持仓
                if current_price >= trade['stop_loss'] or current_price <= trade['take_profit'] or holding_time > self.params['holding_period_max']:

                    # 平仓
                    pnl = (trade['entry_price'] - current_price) * trade['quantity']

                    if current_price >= trade['stop_loss']:
                        reason = "止损"
                        self.losses += 1
                        self.heat_counter += 1
                    elif current_price <= trade['take_profit']:
                        reason = "止盈"
                        self.wins += 1
                        self.heat_counter = max(0, self.heat_counter - 1)
                    else:
                        reason = "时间止损"
                        self.heat_counter += 0.5

                    self.inventory += trade['quantity']
                    self.total_pnl += pnl

                    logging.info(f"🔴 平仓 {trade['quantity']:.6f} ETH @ ${current_price:.2f} ({reason})")
                    logging.info(f"💰 盈亏: ${pnl:.2f}, 总盈亏: ${self.total_pnl:.2f}")

                    trade['closed'] = True
                    trade['exit_price'] = current_price
                    trade['exit_time'] = datetime.now()
                    trade['pnl'] = pnl
                    trade['exit_reason'] = reason

    def calculate_performance_metrics(self):
        """计算性能指标"""
        total_trades = self.wins + self.losses

        if total_trades == 0:
            return {
                'win_rate': 0,
                'total_trades': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'profit_factor': 0
            }

        win_rate = self.wins / total_trades
        avg_pnl = self.total_pnl / total_trades if total_trades > 0 else 0

        # 计算盈亏比
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) <= 0]

        total_wins = sum(t.get('pnl', 0) for t in winning_trades)
        total_losses = abs(sum(t.get('pnl', 0) for t in losing_trades))

        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        return {
            'win_rate': win_rate,
            'total_trades': total_trades,
            'total_pnl': self.total_pnl,
            'avg_pnl': avg_pnl,
            'profit_factor': profit_factor,
            'wins': self.wins,
            'losses': self.losses,
            'inventory': self.inventory,
            'heat_counter': self.heat_counter
        }

    def run_trading_session(self, duration_minutes=30):
        """运行交易会话"""
        logging.info(f"🚀 开始 {duration_minutes} 分钟超高胜率交易会话")
        logging.info("="*60)

        start_time = datetime.now()
        session_end = start_time + timedelta(minutes=duration_minutes)

        while datetime.now() < session_end:
            try:
                # 获取订单簿数据
                lob_data = self.get_order_book()
                if not lob_data:
                    time.sleep(0.1)
                    continue

                # 存储数据
                self.order_book_buffer.append(lob_data)
                self.price_history.append(lob_data['mid_price'])

                # 生成交易信号
                signal, signal_strength = self.generate_trading_signal(lob_data)

                if signal != 0:
                    logging.info(f"📊 信号生成: {signal} (强度: {signal_strength:.2f})")

                    # 判断是否执行交易
                    if self.should_execute_trade(signal, signal_strength, lob_data):
                        self.execute_trade(signal, lob_data)

                # 监控现有持仓
                self.monitor_positions(lob_data)

                # 定期显示性能指标
                if int((datetime.now() - start_time).total_seconds()) % 60 == 0:
                    metrics = self.calculate_performance_metrics()
                    logging.info(f"⏰ 性能更新: 胜率={metrics['win_rate']:.1%}, "
                               f"交易={metrics['total_trades']}, "
                               f"盈亏=${metrics['total_pnl']:.2f}, "
                               f"热度={metrics['heat_counter']}")

                # 高频交易循环
                time.sleep(0.1)  # 100ms

            except KeyboardInterrupt:
                logging.info("🛑 用户手动停止交易会话")
                break
            except Exception as e:
                logging.error(f"❌ 交易循环错误: {e}")
                time.sleep(1)

        # 最终统计
        final_metrics = self.calculate_performance_metrics()
        logging.info("="*60)
        logging.info("🏁 交易会话结束！")
        logging.info(f"📊 最终性能:")
        logging.info(f"   胜率: {final_metrics['win_rate']:.1%} (目标: 80%+)")
        logging.info(f"   总交易: {final_metrics['total_trades']}")
        logging.info(f"   盈亏: {final_metrics['wins']}/{final_metrics['losses']}")
        logging.info(f"   总盈亏: ${final_metrics['total_pnl']:.2f}")
        logging.info(f"   平均盈亏: ${final_metrics['avg_pnl']:.2f}")
        logging.info(f"   盈亏比: {final_metrics['profit_factor']:.2f}")
        logging.info(f"   当前持仓: {final_metrics['inventory']:.6f} ETH")

        # 评估是否达到目标
        if final_metrics['win_rate'] >= 0.8:
            logging.info("🎉 恭喜！达到80%+胜率目标！")
        elif final_metrics['win_rate'] >= 0.7:
            logging.info("🟡 接近目标，胜率70%+，继续优化...")
        else:
            logging.info("🔴 未达到目标，需要进一步优化策略")

        return final_metrics

def main():
    """主函数"""
    try:
        # 创建超高胜率交易系统
        trader = UltraHighWinrateScalper(initial_balance=10000)

        # 运行30分钟交易会话
        results = trader.run_trading_session(duration_minutes=30)

        # 保存结果
        results_file = 'ultra_high_winrate_results.json'
        save_data = {
            'session_time': datetime.now().isoformat(),
            'parameters': trader.params,
            'performance': results,
            'trades': [
                {
                    'timestamp': t['timestamp'].isoformat(),
                    'type': t['type'],
                    'quantity': t['quantity'],
                    'entry_price': t['entry_price'],
                    'pnl': t.get('pnl', 0),
                    'exit_reason': t.get('exit_reason', 'open')
                }
                for t in trader.trades
            ]
        }

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        logging.info(f"📁 详细结果已保存到 {results_file}")

    except Exception as e:
        logging.error(f"主程序运行失败: {e}")

if __name__ == "__main__":
    main()