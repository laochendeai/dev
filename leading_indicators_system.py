#!/usr/bin/env python3
"""
领先指标系统设计
专注于减少滞后性，增强预测能力
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class LeadingIndicatorsSystem:
    def __init__(self):
        # 指标权重配置 - 优先使用领先指标
        self.indicator_weights = {
            'price_momentum': 0.25,      # 价格动量 (最强领先)
            'volume_analysis': 0.20,     # 成交量分析
            'rsi_divergence': 0.15,      # RSI背离 (领先信号)
            'microstructure': 0.15,      # 微观结构分析
            'adaptive_ma': 0.10,         # 自适应移动平均
            'trend_strength': 0.10,      # 趋势强度
            'volatility_regime': 0.05    # 波动率状态
        }

    def calculate_price_momentum_leading(self, prices: List[float], returns: List[float]) -> Dict:
        """
        计算领先价格动量指标
        使用多时间框架动量和加速度分析
        """
        if len(prices) < 10:
            return {'momentum_score': 0, 'acceleration': 0, 'momentum_regime': 'neutral'}

        # 短期动量 (1-3周期)
        short_momentum = np.mean(returns[-3:]) if len(returns) >= 3 else 0

        # 中期动量 (5-10周期)
        medium_momentum = np.mean(returns[-10:]) if len(returns) >= 10 else 0

        # 动量加速度 (动量的变化率)
        if len(returns) >= 6:
            momentum_1 = np.mean(returns[-3:])
            momentum_2 = np.mean(returns[-6:-3])
            acceleration = momentum_1 - momentum_2
        else:
            acceleration = 0

        # 动量一致性 (多个时间框架同向)
        momentum_consistency = 0
        if short_momentum > 0 and medium_momentum > 0:
            momentum_consistency = 1
        elif short_momentum < 0 and medium_momentum < 0:
            momentum_consistency = -1

        # 综合动量评分
        momentum_score = (
            short_momentum * 0.4 +
            medium_momentum * 0.3 +
            acceleration * 0.2 +
            momentum_consistency * 0.1
        )

        # 动量状态分类
        if momentum_score > 0.01:
            momentum_regime = 'strong_bullish'
        elif momentum_score > 0.003:
            momentum_regime = 'bullish'
        elif momentum_score < -0.01:
            momentum_regime = 'strong_bearish'
        elif momentum_score < -0.003:
            momentum_regime = 'bearish'
        else:
            momentum_regime = 'neutral'

        return {
            'momentum_score': momentum_score,
            'acceleration': acceleration,
            'momentum_regime': momentum_regime,
            'short_momentum': short_momentum,
            'medium_momentum': medium_momentum
        }

    def detect_rsi_divergence(self, prices: List[float], rsi_values: List[float]) -> Dict:
        """
        检测RSI背离 - 强烈的领先信号
        """
        if len(prices) < 10 or len(rsi_values) < 10:
            return {'divergence_type': None, 'strength': 0}

        # 获取最近的价格和RSI高低点
        recent_prices = prices[-10:]
        recent_rsi = rsi_values[-10:]

        # 寻找价格高点
        price_highs = []
        rsi_at_price_highs = []

        for i in range(2, len(recent_prices) - 2):
            if (recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1] and
                recent_prices[i] > recent_prices[i-2] and recent_prices[i] > recent_prices[i+2]):
                price_highs.append(recent_prices[i])
                rsi_at_price_highs.append(recent_rsi[i])

        # 寻找价格低点
        price_lows = []
        rsi_at_price_lows = []

        for i in range(2, len(recent_prices) - 2):
            if (recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1] and
                recent_prices[i] < recent_prices[i-2] and recent_prices[i] < recent_prices[i+2]):
                price_lows.append(recent_prices[i])
                rsi_at_price_lows.append(recent_rsi[i])

        # 检测看跌背离 (价格新高，RSI不新高)
        bearish_divergence = False
        bearish_strength = 0

        if len(price_highs) >= 2 and len(rsi_at_price_highs) >= 2:
            if (price_highs[-1] > price_highs[-2] and
                rsi_at_price_highs[-1] < rsi_at_price_highs[-2]):
                bearish_divergence = True
                price_diff = (price_highs[-1] - price_highs[-2]) / price_highs[-2]
                rsi_diff = rsi_at_price_highs[-2] - rsi_at_price_highs[-1]
                bearish_strength = min(price_diff * 100, rsi_diff) / 100

        # 检测看涨背离 (价格新低，RSI不新低)
        bullish_divergence = False
        bullish_strength = 0

        if len(price_lows) >= 2 and len(rsi_at_price_lows) >= 2:
            if (price_lows[-1] < price_lows[-2] and
                rsi_at_price_lows[-1] > rsi_at_price_lows[-2]):
                bullish_divergence = True
                price_diff = (price_lows[-2] - price_lows[-1]) / price_lows[-2]
                rsi_diff = rsi_at_price_lows[-1] - rsi_at_price_lows[-2]
                bullish_strength = min(price_diff * 100, rsi_diff) / 100

        if bearish_divergence:
            return {
                'divergence_type': 'bearish',
                'strength': bearish_strength,
                'signal_strength': 'strong' if bearish_strength > 0.02 else 'moderate'
            }
        elif bullish_divergence:
            return {
                'divergence_type': 'bullish',
                'strength': bullish_strength,
                'signal_strength': 'strong' if bullish_strength > 0.02 else 'moderate'
            }
        else:
            return {
                'divergence_type': None,
                'strength': 0,
                'signal_strength': 'none'
            }

    def calculate_volume_based_signals(self, volumes: List[float], prices: List[float]) -> Dict:
        """
        基于成交量的领先信号分析
        """
        if len(volumes) < 5 or len(prices) < 5:
            return {'volume_signal': 'neutral', 'volume_trend': 0, 'price_volume_sync': 0}

        # 成交量移动平均
        volume_ma5 = np.mean(volumes[-5:])
        volume_ma10 = np.mean(volumes[-10:]) if len(volumes) >= 10 else volume_ma5

        # 当前成交量相对水平
        current_volume = volumes[-1]
        volume_ratio = current_volume / volume_ma5

        # 成交量趋势
        volume_trend = (volume_ma5 - volume_ma10) / volume_ma10 if volume_ma10 > 0 else 0

        # 价格-成交量同步性
        price_change = (prices[-1] - prices[-2]) / prices[-2]
        volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] > 0 else 0

        price_volume_sync = 0
        if price_change > 0 and volume_change > 0:
            price_volume_sync = 1  # 量价齐升
        elif price_change < 0 and volume_change > 0:
            price_volume_sync = -0.5  # 下跌放量
        elif price_change > 0 and volume_change < 0:
            price_volume_sync = -0.3  # 上涨缩量

        # 成交量信号
        if volume_ratio > 1.5 and price_volume_sync > 0:
            volume_signal = 'strong_accumulation'
        elif volume_ratio > 1.2 and price_volume_sync > 0:
            volume_signal = 'accumulation'
        elif volume_ratio > 1.5 and price_volume_sync < 0:
            volume_signal = 'distribution'
        else:
            volume_signal = 'neutral'

        return {
            'volume_signal': volume_signal,
            'volume_ratio': volume_ratio,
            'volume_trend': volume_trend,
            'price_volume_sync': price_volume_sync
        }

    def calculate_microstructure_signals(self, price_data: List[Dict]) -> Dict:
        """
        微观结构分析 - 基于价格波动的领先指标
        """
        if len(price_data) < 5:
            return {'microstructure_signal': 'neutral', 'volatility_regime': 'normal'}

        # 计算价格跳动模式
        price_changes = [data['change'] for data in price_data[-5:]]

        # 价格连续性
        consecutive_up = sum(1 for i in range(len(price_changes)-1)
                           if price_changes[i] > 0 and price_changes[i+1] > 0)
        consecutive_down = sum(1 for i in range(len(price_changes)-1)
                             if price_changes[i] < 0 and price_changes[i+1] < 0)

        # 波动率状态
        volatility = np.std(price_changes) if len(price_changes) > 1 else 0
        avg_price = np.mean([data['price'] for data in price_data[-5:]])
        volatility_pct = (volatility / avg_price) * 100 if avg_price > 0 else 0

        # 微观结构信号
        microstructure_signal = 'neutral'
        if consecutive_up >= 3 and volatility_pct < 0.5:
            microstructure_signal = 'steady_rise'
        elif consecutive_down >= 3 and volatility_pct < 0.5:
            microstructure_signal = 'steady_decline'
        elif volatility_pct > 2.0:
            microstructure_signal = 'high_volatility'
        elif abs(np.mean(price_changes)) < 0.001:
            microstructure_signal = 'consolidation'

        # 波动率状态
        if volatility_pct < 0.3:
            volatility_regime = 'low'
        elif volatility_pct < 1.0:
            volatility_regime = 'normal'
        elif volatility_pct < 2.0:
            volatility_regime = 'elevated'
        else:
            volatility_regime = 'high'

        return {
            'microstructure_signal': microstructure_signal,
            'volatility_regime': volatility_regime,
            'consecutive_moves': max(consecutive_up, consecutive_down),
            'volatility_pct': volatility_pct
        }

    def calculate_adaptive_ma_signals(self, prices: List[float]) -> Dict:
        """
        自适应移动平均 - 减少滞后性
        """
        if len(prices) < 10:
            return {'adaptive_signal': 'neutral', 'trend_strength': 0}

        # 计算效率比率 (Efficiency Ratio)
        if len(prices) >= 10:
            price_change = abs(prices[-1] - prices[-10])
            total_movement = sum(abs(prices[i] - prices[i-1]) for i in range(-9, 0))
            efficiency_ratio = price_change / total_movement if total_movement > 0 else 0
        else:
            efficiency_ratio = 0.5

        # 自适应平滑常数
        fast_sc = 2 / (3 + 1)  # 3周期快速
        slow_sc = 2 / (30 + 1)  # 30周期慢速
        smoothing_constant = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc)

        # 计算自适应移动平均
        adaptive_ma = prices[-10]
        for price in prices[-9:]:
            adaptive_ma = adaptive_ma + smoothing_constant * (price - adaptive_ma)

        # 趋势强度
        trend_strength = (prices[-1] - adaptive_ma) / adaptive_ma

        # 自适应信号
        if trend_strength > 0.01 and efficiency_ratio > 0.5:
            adaptive_signal = 'strong_uptrend'
        elif trend_strength > 0.003 and efficiency_ratio > 0.3:
            adaptive_signal = 'uptrend'
        elif trend_strength < -0.01 and efficiency_ratio > 0.5:
            adaptive_signal = 'strong_downtrend'
        elif trend_strength < -0.003 and efficiency_ratio > 0.3:
            adaptive_signal = 'downtrend'
        else:
            adaptive_signal = 'neutral'

        return {
            'adaptive_signal': adaptive_signal,
            'trend_strength': trend_strength,
            'efficiency_ratio': efficiency_ratio,
            'adaptive_ma': adaptive_ma
        }

    def generate_leading_signals(self, data: Dict) -> Dict:
        """
        综合生成领先交易信号
        """
        prices = data.get('prices', [])
        volumes = data.get('volumes', [])
        rsi_values = data.get('rsi_values', [])

        # 计算各项领先指标
        momentum_analysis = self.calculate_price_momentum_leading(
            prices, data.get('returns', [])
        )

        rsi_divergence = self.detect_rsi_divergence(prices, rsi_values)

        volume_analysis = self.calculate_volume_based_signals(volumes, prices)

        microstructure = self.calculate_microstructure_signals(
            data.get('price_data', [])
        )

        adaptive_ma = self.calculate_adaptive_ma_signals(prices)

        # 综合信号评分
        buy_score = 0
        sell_score = 0

        # 价格动量 (权重: 25%)
        if momentum_analysis['momentum_regime'] in ['bullish', 'strong_bullish']:
            buy_score += self.indicator_weights['price_momentum']
        elif momentum_analysis['momentum_regime'] in ['bearish', 'strong_bearish']:
            sell_score += self.indicator_weights['price_momentum']

        # RSI背离 (权重: 15%)
        if rsi_divergence['divergence_type'] == 'bullish':
            buy_score += self.indicator_weights['rsi_divergence'] * rsi_divergence['strength']
        elif rsi_divergence['divergence_type'] == 'bearish':
            sell_score += self.indicator_weights['rsi_divergence'] * rsi_divergence['strength']

        # 成交量分析 (权重: 20%)
        if volume_analysis['volume_signal'] in ['accumulation', 'strong_accumulation']:
            buy_score += self.indicator_weights['volume_analysis']
        elif volume_analysis['volume_signal'] == 'distribution':
            sell_score += self.indicator_weights['volume_analysis']

        # 自适应移动平均 (权重: 10%)
        if adaptive_ma['adaptive_signal'] in ['uptrend', 'strong_uptrend']:
            buy_score += self.indicator_weights['adaptive_ma']
        elif adaptive_ma['adaptive_signal'] in ['downtrend', 'strong_downtrend']:
            sell_score += self.indicator_weights['adaptive_ma']

        # 微观结构 (权重: 15%)
        if microstructure['microstructure_signal'] == 'steady_rise':
            buy_score += self.indicator_weights['microstructure']
        elif microstructure['microstructure_signal'] == 'steady_decline':
            sell_score += self.indicator_weights['microstructure']

        # 确定最终信号
        signal_strength = abs(buy_score - sell_score)

        if buy_score > sell_score and signal_strength > 0.4:
            final_signal = 'strong_buy' if signal_strength > 0.7 else 'buy'
        elif sell_score > buy_score and signal_strength > 0.4:
            final_signal = 'strong_sell' if signal_strength > 0.7 else 'sell'
        else:
            final_signal = 'hold'

        return {
            'signal': final_signal,
            'signal_strength': signal_strength,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'components': {
                'momentum': momentum_analysis,
                'rsi_divergence': rsi_divergence,
                'volume': volume_analysis,
                'adaptive_ma': adaptive_ma,
                'microstructure': microstructure
            }
        }