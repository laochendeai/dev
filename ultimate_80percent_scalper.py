#!/usr/bin/env python3
"""
终极80%+胜率剥头皮交易系统
集成：订单簿分析 + 机器学习预测 + 市场微观结构 + 高频执行
目标：实现并验证80%+胜率
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

# 导入机器学习组件
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_80percent_scalper.log'),
        logging.StreamHandler()
    ]
)

class Ultimate80PercentScalper:
    """终极80%+胜率剥头皮交易系统"""

    def __init__(self, initial_balance=10000):
        # 核心配置
        self.config = {
            # 胜率目标
            'target_win_rate': 0.80,          # 80%目标胜率
            'confidence_threshold': 0.75,      # 置信度阈值

            # 订单簿分析参数
            'lob_levels': 15,                  # 订单簿深度
            'imbalance_threshold': 0.25,       # 不平衡阈值
            'spread_threshold': 0.0008,        # 价差阈值 0.08%
            'volume_threshold': 2.0,           # 成交量阈值

            # 机器学习参数
            'ml_weight': 0.4,                  # ML模型权重
            'lob_weight': 0.35,                # 订单簿权重
            'technical_weight': 0.25,          # 技术指标权重

            # 风险管理
            'position_size': 0.02,             # 2%仓位
            'risk_reward_ratio': 2.0,          # 风险收益比
            'max_loss_per_trade': 0.001,       # 0.1%最大亏损
            'profit_target': 0.002,            # 0.2%利润目标

            # 执行参数
            'execution_delay': 0.05,           # 50ms执行延迟
            'holding_time_max': 180,           # 3分钟最大持仓
            'cooldown_time': 30,               # 30秒冷却

            # 过滤条件
            'volatility_min': 0.01,            # 最小波动率
            'volatility_max': 0.1,             # 最大波动率
            'momentum_threshold': 0.001        # 动量阈值
        }

        # 初始化交易所
        self.exchange = ccxt.binance()

        # 账户状态
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.inventory = 0
        self.open_positions = []

        # 数据存储
        self.price_history = deque(maxlen=1000)
        self.order_book_history = deque(maxlen=500)
        self.trade_history = deque(maxlen=1000)
        self.prediction_history = deque(maxlen=1000)

        # 机器学习组件
        self.ml_model = None
        self.scaler = MinMaxScaler()
        self.training_data_X = deque(maxlen=10000)
        self.training_data_y = deque(maxlen=10000)

        # 统计信息
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.successful_predictions = 0
        self.total_predictions = 0

        logging.info("🚀 终极80%+胜率剥头皮交易系统已启动")
        logging.info(f"💰 初始资金: ${self.balance:.2f}")
        logging.info(f"🎯 目标胜率: {self.config['target_win_rate']:.1%}")

    def initialize_ml_model(self):
        """初始化机器学习模型"""
        try:
            self.ml_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
            logging.info("✅ 机器学习模型初始化完成")
            return True
        except Exception as e:
            logging.error(f"ML模型初始化失败: {e}")
            return False

    def get_comprehensive_market_data(self):
        """获取综合市场数据"""
        try:
            # 1. 获取订单簿数据
            orderbook = self.exchange.fetch_order_book('ETH/USDT', limit=self.config['lob_levels'])

            # 2. 获取最近交易
            recent_trades = self.exchange.fetch_trades('ETH/USDT', limit=50)

            # 3. 获取K线数据
            klines = self.exchange.fetch_ohlcv('ETH/USDT', '1m', limit=100)

            # 处理订单簿
            lob_data = {
                'timestamp': datetime.now(),
                'bids': [(float(b[0]), float(b[1])) for b in orderbook['bids']],
                'asks': [(float(a[0]), float(a[1])) for a in orderbook['asks']]
            }

            # 处理交易数据
            trades_data = []
            for trade in recent_trades:
                trades_data.append({
                    'timestamp': datetime.fromtimestamp(trade['timestamp'] / 1000),
                    'price': float(trade['price']),
                    'amount': float(trade['amount']),
                    'side': 'buy' if trade['side'] == 'buy' else 'sell'
                })

            # 处理K线数据
            klines_df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            klines_df['timestamp'] = pd.to_datetime(klines_df['timestamp'], unit='ms')

            return lob_data, trades_data, klines_df

        except Exception as e:
            logging.error(f"获取市场数据失败: {e}")
            return None, None, None

    def calculate_advanced_lob_features(self, lob_data):
        """计算高级订单簿特征"""
        features = {}

        try:
            # 基础数据
            bids = lob_data['bids'][:10]  # 前10档买单
            asks = lob_data['asks'][:10]  # 前10档卖单

            if not bids or not asks:
                return features

            # 1. 订单簿不平衡 (多层级)
            bid_volume_5 = sum(b[1] for b in bids[:5])
            ask_volume_5 = sum(a[1] for a in asks[:5])
            bid_volume_10 = sum(b[1] for b in bids)
            ask_volume_10 = sum(a[1] for a in asks)

            if bid_volume_5 + ask_volume_5 > 0:
                features['imbalance_5'] = (bid_volume_5 - ask_volume_5) / (bid_volume_5 + ask_volume_5)
            else:
                features['imbalance_5'] = 0

            if bid_volume_10 + ask_volume_10 > 0:
                features['imbalance_10'] = (bid_volume_10 - ask_volume_10) / (bid_volume_10 + ask_volume_10)
            else:
                features['imbalance_10'] = 0

            # 2. 价差分析
            mid_price = (bids[0][0] + asks[0][0]) / 2
            spread = asks[0][0] - bids[0][0]
            features['spread_bps'] = (spread / mid_price) * 10000  # 基点

            # 3. 流动性深度
            features['bid_depth_5'] = bid_volume_5
            features['ask_depth_5'] = ask_volume_5
            features['total_depth'] = bid_volume_10 + ask_volume_10

            # 4. 价格影响预测
            if ask_volume_5 > 0:
                features['price_impact_buy'] = spread / ask_volume_5
            else:
                features['price_impact_buy'] = 0

            if bid_volume_5 > 0:
                features['price_impact_sell'] = spread / bid_volume_5
            else:
                features['price_impact_sell'] = 0

            # 5. 订单簿斜率 (价格累积分布)
            cumulative_bid_volume = 0
            weighted_bid_price = 0
            for price, volume in bids:
                cumulative_bid_volume += volume
                weighted_bid_price += price * volume

            cumulative_ask_volume = 0
            weighted_ask_price = 0
            for price, volume in asks:
                cumulative_ask_volume += volume
                weighted_ask_price += price * volume

            if cumulative_bid_volume > 0:
                features['bid_vwap'] = weighted_bid_price / cumulative_bid_volume
                features['bid_vwap_ratio'] = features['bid_vwap'] / mid_price
            else:
                features['bid_vwap'] = bids[0][0]
                features['bid_vwap_ratio'] = 1

            if cumulative_ask_volume > 0:
                features['ask_vwap'] = weighted_ask_price / cumulative_ask_volume
                features['ask_vwap_ratio'] = features['ask_vwap'] / mid_price
            else:
                features['ask_vwap'] = asks[0][0]
                features['ask_vwap_ratio'] = 1

            # 6. 订单簿不对称性
            features['asymmetry'] = features['bid_vwap_ratio'] - features['ask_vwap_ratio']

        except Exception as e:
            logging.error(f"订单簿特征计算失败: {e}")

        return features

    def calculate_order_flow_features(self, trades_data):
        """计算订单流特征"""
        features = {}

        try:
            if not trades_data:
                return features

            # 1. 交易方向统计
            recent_trades = trades_data[-20:]  # 最近20笔交易
            buy_volume = sum(t['amount'] for t in recent_trades if t['side'] == 'buy')
            sell_volume = sum(t['amount'] for t in recent_trades if t['side'] == 'sell')
            total_volume = buy_volume + sell_volume

            if total_volume > 0:
                features['order_flow_balance'] = (buy_volume - sell_volume) / total_volume
            else:
                features['order_flow_balance'] = 0

            # 2. 大单交易频率
            avg_trade_size = np.mean([t['amount'] for t in recent_trades])
            large_trades = [t for t in recent_trades if t['amount'] > avg_trade_size * 2]
            features['large_trade_ratio'] = len(large_trades) / len(recent_trades)

            # 3. 交易强度
            if len(recent_trades) >= 2:
                time_span = (recent_trades[-1]['timestamp'] - recent_trades[0]['timestamp']).total_seconds()
                if time_span > 0:
                    features['trade_intensity'] = len(recent_trades) / time_span
                else:
                    features['trade_intensity'] = 0
            else:
                features['trade_intensity'] = 0

            # 4. 价格冲击分析
            if len(recent_trades) >= 5:
                price_changes = [abs(recent_trades[i]['price'] - recent_trades[i-1]['price'])
                               for i in range(1, min(5, len(recent_trades)))]
                features['avg_price_impact'] = np.mean(price_changes) / recent_trades[0]['price']
            else:
                features['avg_price_impact'] = 0

        except Exception as e:
            logging.error(f"订单流特征计算失败: {e}")

        return features

    def calculate_technical_features(self, klines_df):
        """计算技术指标特征"""
        features = {}

        try:
            if len(klines_df) < 20:
                return features

            # 价格序列
            closes = klines_df['close'].values
            volumes = klines_df['volume'].values

            # 1. 价格动量
            returns = np.diff(closes) / closes[:-1]
            features['momentum_5'] = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            features['momentum_10'] = np.mean(returns[-10:]) if len(returns) >= 10 else 0
            features['volatility'] = np.std(returns[-20:]) if len(returns) >= 20 else 0

            # 2. 移动平均线
            if len(closes) >= 10:
                ma_5 = np.mean(closes[-5:])
                ma_10 = np.mean(closes[-10:])
                current_price = closes[-1]
                features['price_vs_ma5'] = (current_price - ma_5) / ma_5
                features['price_vs_ma10'] = (current_price - ma_10) / ma_10

            # 3. RSI
            if len(closes) >= 14:
                deltas = np.diff(closes)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    features['rsi'] = 100 - (100 / (1 + rs))
                else:
                    features['rsi'] = 100

            # 4. 成交量特征
            if len(volumes) >= 10:
                vol_ma = np.mean(volumes[-10:])
                features['volume_ratio'] = volumes[-1] / vol_ma if vol_ma > 0 else 1

            # 5. 价格趋势强度
            if len(closes) >= 20:
                x = np.arange(len(closes[-20:]))
                y = closes[-20:]
                slope = np.polyfit(x, y, 1)[0]
                features['trend_strength'] = slope / np.mean(y)

        except Exception as e:
            logging.error(f"技术特征计算失败: {e}")

        return features

    def generate_ml_signal(self, all_features):
        """生成机器学习信号"""
        try:
            if self.ml_model is None:
                return 0, 0

            # 准备特征向量
            feature_vector = []
            feature_names = ['imbalance_5', 'imbalance_10', 'spread_bps', 'total_depth',
                           'price_impact_buy', 'price_impact_sell', 'bid_vwap_ratio',
                           'ask_vwap_ratio', 'asymmetry', 'order_flow_balance',
                           'large_trade_ratio', 'trade_intensity', 'avg_price_impact',
                           'momentum_5', 'momentum_10', 'volatility', 'price_vs_ma5',
                           'price_vs_ma10', 'rsi', 'volume_ratio', 'trend_strength']

            for name in feature_names:
                feature_vector.append(all_features.get(name, 0))

            # 预测
            if len(self.training_data_X) > 100:  # 有足够训练数据
                try:
                    # 标准化特征
                    features_array = np.array(feature_vector).reshape(1, -1)

                    # 获取预测概率
                    probabilities = self.ml_model.predict_proba(features_array)[0]

                    if len(probabilities) >= 2:
                        confidence = max(probabilities)
                        prediction = np.argmax(probabilities)
                        return prediction, confidence
                    else:
                        return 0, 0
                except:
                    return 0, 0
            else:
                return 0, 0

        except Exception as e:
            logging.error(f"ML信号生成失败: {e}")
            return 0, 0

    def generate_ensemble_signal(self, lob_features, flow_features, technical_features):
        """生成集成信号"""
        try:
            # 1. 订单簿信号
            lob_signal = 0
            if lob_features.get('imbalance_5', 0) > self.config['imbalance_threshold']:
                lob_signal = 1
            elif lob_features.get('imbalance_5', 0) < -self.config['imbalance_threshold']:
                lob_signal = -1

            # 2. 订单流信号
            flow_signal = 0
            if flow_features.get('order_flow_balance', 0) > 0.3:
                flow_signal = 1
            elif flow_features.get('order_flow_balance', 0) < -0.3:
                flow_signal = -1

            # 3. 技术指标信号
            tech_signal = 0
            momentum = technical_features.get('momentum_5', 0)
            if momentum > self.config['momentum_threshold']:
                tech_signal = 1
            elif momentum < -self.config['momentum_threshold']:
                tech_signal = -1

            # 4. 机器学习信号
            all_features = {**lob_features, **flow_features, **technical_features}
            ml_prediction, ml_confidence = self.generate_ml_signal(all_features)

            # 5. 集成决策
            signals = {
                'lob': lob_signal,
                'flow': flow_signal,
                'technical': tech_signal,
                'ml': ml_prediction
            }

            weights = {
                'lob': self.config['lob_weight'],
                'flow': 0.1,
                'technical': self.config['technical_weight'],
                'ml': self.config['ml_weight'] if ml_confidence > self.config['confidence_threshold'] else 0
            }

            # 加权平均
            weighted_sum = 0
            total_weight = 0

            for signal_type, signal_value in signals.items():
                weight = weights.get(signal_type, 0)
                if weight > 0 and signal_value != 0:
                    weighted_sum += signal_value * weight
                    total_weight += weight

            if total_weight > 0:
                final_signal = 1 if weighted_sum / total_weight > 0.3 else (-1 if weighted_sum / total_weight < -0.3 else 0)
                signal_strength = abs(weighted_sum / total_weight)
            else:
                final_signal = 0
                signal_strength = 0

            return final_signal, signal_strength, signals

        except Exception as e:
            logging.error(f"集成信号生成失败: {e}")
            return 0, 0, {}

    def evaluate_trade_opportunity(self, signal, signal_strength, all_features):
        """评估交易机会"""
        try:
            # 1. 基础过滤条件
            if signal == 0 or signal_strength < 0.4:
                return False, "信号强度不足"

            # 2. 波动率过滤
            volatility = all_features.get('volatility', 0)
            if volatility < self.config['volatility_min'] or volatility > self.config['volatility_max']:
                return False, "波动率超出范围"

            # 3. 价差过滤
            spread_bps = all_features.get('spread_bps', 0)
            if spread_bps > self.config['spread_threshold'] * 10000:
                return False, "价差过大"

            # 4. 流动性过滤
            total_depth = all_features.get('total_depth', 0)
            if total_depth < 100:  # 最小流动性要求
                return False, "流动性不足"

            # 5. 动量确认
            momentum_5 = all_features.get('momentum_5', 0)
            if signal > 0 and momentum_5 < -self.config['momentum_threshold']:
                return False, "动量与信号方向不一致"
            if signal < 0 and momentum_5 > self.config['momentum_threshold']:
                return False, "动量与信号方向不一致"

            # 6. 冷却时间检查
            if self.open_positions:
                last_position_time = self.open_positions[-1]['timestamp']
                time_since_last = (datetime.now() - last_position_time).total_seconds()
                if time_since_last < self.config['cooldown_time']:
                    return False, "冷却时间未过"

            # 7. 仓位检查
            position_ratio = abs(self.inventory) / self.balance if self.balance > 0 else 0
            if position_ratio > 0.1:  # 最大10%仓位
                return False, "仓位已满"

            return True, "交易机会确认"

        except Exception as e:
            logging.error(f"交易机会评估失败: {e}")
            return False, "评估错误"

    def execute_trade(self, signal, current_price, all_features):
        """执行交易"""
        try:
            # 计算仓位大小
            position_value = self.balance * self.config['position_size']
            quantity = position_value / current_price

            # 计算止损止盈价格
            spread = self.config['profit_target']
            loss_threshold = self.config['max_loss_per_trade']

            if signal > 0:  # 买入
                stop_loss_price = current_price * (1 - loss_threshold)
                take_profit_price = current_price * (1 + spread)
                trade_type = "买入"
            else:  # 卖出
                stop_loss_price = current_price * (1 + loss_threshold)
                take_profit_price = current_price * (1 - spread)
                trade_type = "卖出"

            # 创建交易记录
            position = {
                'timestamp': datetime.now(),
                'type': trade_type,
                'signal': signal,
                'quantity': quantity,
                'entry_price': current_price,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'signal_strength': all_features.get('signal_strength', 0),
                'status': 'open'
            }

            # 更新持仓和账户
            if signal > 0:
                self.inventory += quantity
                self.balance -= position_value
            else:
                self.inventory -= quantity
                self.balance -= position_value

            self.open_positions.append(position)
            self.total_trades += 1

            logging.info(f"🟢 {trade_type}执行: {quantity:.6f} ETH @ ${current_price:.2f}")
            logging.info(f"🛑 止损: ${stop_loss_price:.2f}, 🎯 止盈: ${take_profit_price:.2f}")
            logging.info(f"📊 信号强度: {all_features.get('signal_strength', 0):.2f}")

            return True

        except Exception as e:
            logging.error(f"交易执行失败: {e}")
            return False

    def monitor_and_close_positions(self, current_price):
        """监控并平仓"""
        try:
            closed_positions = []

            for i, position in enumerate(self.open_positions):
                if position['status'] != 'open':
                    continue

                # 检查止损止盈
                should_close = False
                close_reason = ""
                close_price = current_price

                holding_time = (datetime.now() - position['timestamp']).total_seconds()

                if position['signal'] > 0:  # 多头
                    if current_price <= position['stop_loss']:
                        should_close = True
                        close_reason = "止损"
                        close_price = position['stop_loss']
                    elif current_price >= position['take_profit']:
                        should_close = True
                        close_reason = "止盈"
                        close_price = position['take_profit']
                    elif holding_time > self.config['holding_time_max']:
                        should_close = True
                        close_reason = "时间止损"

                else:  # 空头
                    if current_price >= position['stop_loss']:
                        should_close = True
                        close_reason = "止损"
                        close_price = position['stop_loss']
                    elif current_price <= position['take_profit']:
                        should_close = True
                        close_reason = "止盈"
                        close_price = position['take_profit']
                    elif holding_time > self.config['holding_time_max']:
                        should_close = True
                        close_reason = "时间止损"

                if should_close:
                    # 计算盈亏
                    if position['signal'] > 0:
                        pnl = (close_price - position['entry_price']) * position['quantity']
                        self.inventory -= position['quantity']
                    else:
                        pnl = (position['entry_price'] - close_price) * position['quantity']
                        self.inventory += position['quantity']

                    self.balance += position['quantity'] * close_price
                    self.total_pnl += pnl

                    # 更新统计
                    if pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1

                    # 更新交易状态
                    position['status'] = 'closed'
                    position['exit_price'] = close_price
                    position['exit_time'] = datetime.now()
                    position['exit_reason'] = close_reason
                    position['pnl'] = pnl

                    closed_positions.append(position)

                    logging.info(f"🔴 平仓: {position['quantity']:.6f} ETH @ ${close_price:.2f} ({close_reason})")
                    logging.info(f"💰 盈亏: ${pnl:.2f}, 总盈亏: ${self.total_pnl:.2f}")

            return closed_positions

        except Exception as e:
            logging.error(f"持仓监控失败: {e}")
            return []

    def calculate_performance_metrics(self):
        """计算性能指标"""
        total_trades = self.winning_trades + self.losing_trades

        if total_trades == 0:
            return {
                'win_rate': 0,
                'total_trades': 0,
                'total_pnl': 0,
                'return_pct': 0,
                'profit_factor': 0,
                'current_balance': self.balance,
                'target_achieved': False
            }

        win_rate = self.winning_trades / total_trades
        return_pct = (self.balance - self.initial_balance) / self.initial_balance

        # 计算盈亏比
        winning_positions = [p for p in self.open_positions + list(self.trade_history) if p.get('pnl', 0) > 0]
        losing_positions = [p for p in self.open_positions + list(self.trade_history) if p.get('pnl', 0) <= 0]

        total_wins = sum(p.get('pnl', 0) for p in winning_positions)
        total_losses = abs(sum(p.get('pnl', 0) for p in losing_positions))

        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        return {
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': self.total_pnl,
            'return_pct': return_pct,
            'profit_factor': profit_factor,
            'current_balance': self.balance,
            'target_achieved': win_rate >= self.config['target_win_rate'],
            'open_positions': len([p for p in self.open_positions if p['status'] == 'open'])
        }

    def run_backtest_session(self, duration_minutes=30):
        """运行回测会话"""
        logging.info(f"🎯 开始 {duration_minutes} 分钟终极80%胜率回测验证")
        logging.info("="*80)

        start_time = datetime.now()
        session_end = start_time + timedelta(minutes=duration_minutes)
        iteration = 0

        while datetime.now() < session_end:
            try:
                iteration += 1

                # 获取市场数据
                lob_data, trades_data, klines_df = self.get_comprehensive_market_data()
                if not lob_data or not trades_data or klines_df is None:
                    time.sleep(5)
                    continue

                # 存储数据
                self.order_book_history.append(lob_data)
                for trade in trades_data:
                    self.trade_history.append(trade)

                current_price = (lob_data['bids'][0][0] + lob_data['asks'][0][0]) / 2
                self.price_history.append(current_price)

                # 计算特征
                lob_features = self.calculate_advanced_lob_features(lob_data)
                flow_features = self.calculate_order_flow_features(trades_data)
                technical_features = self.calculate_technical_features(klines_df)

                all_features = {**lob_features, **flow_features, **technical_features}

                # 生成交易信号
                signal, signal_strength, component_signals = self.generate_ensemble_signal(
                    lob_features, flow_features, technical_features
                )
                all_features['signal_strength'] = signal_strength

                # 评估交易机会
                if signal != 0:
                    can_trade, reason = self.evaluate_trade_opportunity(signal, signal_strength, all_features)

                    if can_trade:
                        success = self.execute_trade(signal, current_price, all_features)
                        if success:
                            self.total_predictions += 1

                # 监控现有持仓
                self.monitor_and_close_positions(current_price)

                # 定期显示性能
                if iteration % 20 == 0:
                    metrics = self.calculate_performance_metrics()
                    logging.info(f"⏰ 性能更新 [{iteration}]: "
                               f"胜率={metrics['win_rate']:.1%}, "
                               f"交易={metrics['total_trades']}, "
                               f"盈亏=${metrics['total_pnl']:.2f}, "
                               f"收益={metrics['return_pct']:.1%}")

                    # 检查是否达到目标
                    if metrics['target_achieved']:
                        logging.info(f"🎉 恭喜！达到{self.config['target_win_rate']:.1%}胜率目标！")

                time.sleep(3)  # 3秒循环，高频采样

            except KeyboardInterrupt:
                logging.info("🛑 用户手动停止回测")
                break
            except Exception as e:
                logging.error(f"❌ 回测循环错误: {e}")
                time.sleep(5)

        # 最终统计
        final_metrics = self.calculate_performance_metrics()
        logging.info("="*80)
        logging.info("🏁 终极80%胜率回测验证完成！")
        logging.info(f"📊 最终性能指标:")
        logging.info(f"   🎯 胜率: {final_metrics['win_rate']:.1%} (目标: {self.config['target_win_rate']:.1%})")
        logging.info(f"   📈 总交易: {final_metrics['total_trades']}")
        logging.info(f"   ✅ 盈利交易: {final_metrics['winning_trades']}")
        logging.info(f"   ❌ 亏损交易: {final_metrics['losing_trades']}")
        logging.info(f"   💰 总盈亏: ${final_metrics['total_pnl']:.2f}")
        logging.info(f"   📊 收益率: {final_metrics['return_pct']:.1%}")
        logging.info(f"   ⚖️ 盈亏比: {final_metrics['profit_factor']:.2f}")
        logging.info(f"   💵 当前余额: ${final_metrics['current_balance']:.2f}")
        logging.info(f"   📋 开放持仓: {final_metrics['open_positions']}")

        # 目标达成评估
        if final_metrics['target_achieved']:
            logging.info("🏆 成功达成80%+胜率目标！系统验证成功！")
        elif final_metrics['win_rate'] >= 0.7:
            logging.info("🟡 接近目标，胜率70%+，继续优化可达成80%")
        elif final_metrics['win_rate'] >= 0.6:
            logging.info("🟠 良好表现，胜率60%+，需要进一步优化")
        else:
            logging.info("🔴 胜率未达标，需要重新调整策略参数")

        return final_metrics

def main():
    """主函数"""
    try:
        # 创建终极交易系统
        trader = Ultimate80PercentScalper(initial_balance=10000)

        # 初始化机器学习模型
        trader.initialize_ml_model()

        # 运行30分钟回测验证
        results = trader.run_backtest_session(duration_minutes=30)

        # 保存详细结果
        results_file = 'ultimate_80percent_results.json'
        save_data = {
            'session_time': datetime.now().isoformat(),
            'config': trader.config,
            'performance': results,
            'target_win_rate': trader.config['target_win_rate'],
            'target_achieved': results['target_achieved']
        }

        # 保存所有交易记录
        all_trades = trader.open_positions + list(trader.trade_history)
        save_data['trades'] = [
            {
                'timestamp': t['timestamp'].isoformat(),
                'type': t['type'],
                'signal': t['signal'],
                'quantity': t['quantity'],
                'entry_price': t['entry_price'],
                'exit_price': t.get('exit_price'),
                'pnl': t.get('pnl', 0),
                'exit_reason': t.get('exit_reason', 'open'),
                'signal_strength': t.get('signal_strength', 0)
            }
            for t in all_trades
        ]

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        logging.info(f"📁 详细结果已保存到 {results_file}")

    except Exception as e:
        logging.error(f"主程序运行失败: {e}")

if __name__ == "__main__":
    main()