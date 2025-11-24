#!/usr/bin/env python3
"""
信号诊断工具
分析为什么80%胜率系统没有生成交易信号
"""

import ccxt
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

class SignalDiagnosis:
    """信号诊断分析器"""

    def __init__(self):
        self.exchange = ccxt.binance()
        self.config = {
            'imbalance_threshold': 0.25,
            'spread_threshold': 0.0008,
            'volume_threshold': 2.0,
            'momentum_threshold': 0.001,
            'confidence_threshold': 0.75,
            'volatility_min': 0.01,
            'volatility_max': 0.1
        }

    def get_market_data(self):
        """获取市场数据"""
        try:
            # 订单簿
            orderbook = self.exchange.fetch_order_book('ETH/USDT', limit=20)
            lob_data = {
                'bids': [(float(b[0]), float(b[1])) for b in orderbook['bids']],
                'asks': [(float(a[0]), float(a[1])) for a in orderbook['asks']]
            }

            # 交易数据
            recent_trades = self.exchange.fetch_trades('ETH/USDT', limit=50)
            trades_data = []
            for trade in recent_trades:
                trades_data.append({
                    'price': float(trade['price']),
                    'amount': float(trade['amount']),
                    'side': 'buy' if trade['side'] == 'buy' else 'sell'
                })

            # K线数据
            klines = self.exchange.fetch_ohlcv('ETH/USDT', '1m', limit=100)
            klines_df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            return lob_data, trades_data, klines_df

        except Exception as e:
            logging.error(f"获取数据失败: {e}")
            return None, None, None

    def analyze_lob_signals(self, lob_data):
        """分析订单簿信号"""
        print("\n📊 订单簿信号分析:")

        try:
            bids = lob_data['bids'][:10]
            asks = lob_data['asks'][:10]

            # 当前价格
            current_price = (bids[0][0] + asks[0][0]) / 2
            print(f"   当前价格: ${current_price:.2f}")

            # 价差
            spread = asks[0][0] - bids[0][0]
            spread_bps = (spread / current_price) * 10000
            print(f"   价差: {spread:.2f} ({spread_bps:.1f} bps)")
            print(f"   价差阈值: {self.config['spread_threshold'] * 10000:.1f} bps")
            print(f"   价差检查: {'✅ 通过' if spread_bps < self.config['spread_threshold'] * 10000 else '❌ 超出'}")

            # 订单不平衡
            bid_volume_5 = sum(b[1] for b in bids[:5])
            ask_volume_5 = sum(a[1] for a in asks[:5])
            if bid_volume_5 + ask_volume_5 > 0:
                imbalance = (bid_volume_5 - ask_volume_5) / (bid_volume_5 + ask_volume_5)
            else:
                imbalance = 0

            print(f"   买量(5档): {bid_volume_5:.2f}")
            print(f"   卖量(5档): {ask_volume_5:.2f}")
            print(f"   不平衡: {imbalance:.3f}")
            print(f"   不平衡阈值: ±{self.config['imbalance_threshold']}")
            print(f"   不平衡检查: {'✅ 偏多' if imbalance > self.config['imbalance_threshold'] else '✅ 偏空' if imbalance < -self.config['imbalance_threshold'] else '❌ 中性'}")

            # 流动性
            total_depth = bid_volume_5 + ask_volume_5
            print(f"   总深度: {total_depth:.2f}")
            print(f"   流动性检查: {'✅ 充足' if total_depth > 100 else '❌ 不足'}")

            return {
                'price': current_price,
                'spread_bps': spread_bps,
                'imbalance': imbalance,
                'total_depth': total_depth,
                'lob_signal': 1 if imbalance > self.config['imbalance_threshold'] else (-1 if imbalance < -self.config['imbalance_threshold'] else 0)
            }

        except Exception as e:
            logging.error(f"订单簿分析失败: {e}")
            return {}

    def analyze_order_flow_signals(self, trades_data):
        """分析订单流信号"""
        print("\n🔄 订单流信号分析:")

        try:
            if not trades_data:
                print("   ❌ 无交易数据")
                return {}

            recent_trades = trades_data[-20:]
            buy_volume = sum(t['amount'] for t in recent_trades if t['side'] == 'buy')
            sell_volume = sum(t['amount'] for t in recent_trades if t['side'] == 'sell')
            total_volume = buy_volume + sell_volume

            print(f"   最近20笔交易统计:")
            print(f"   买量: {buy_volume:.2f}")
            print(f"   卖量: {sell_volume:.2f}")
            print(f"   总量: {total_volume:.2f}")

            if total_volume > 0:
                balance = (buy_volume - sell_volume) / total_volume
                print(f"   余额: {balance:.3f}")
                print(f"   信号: {'✅ 偏多' if balance > 0.3 else '✅ 偏空' if balance < -0.3 else '❌ 中性'}")
            else:
                balance = 0
                print(f"   余额: {balance:.3f} (无交易)")

            # 交易强度
            if len(recent_trades) >= 2:
                time_span = 60  # 假设1分钟内
                intensity = len(recent_trades) / max(time_span, 1)
                print(f"   交易强度: {intensity:.2f} 笔/秒")

            return {
                'order_flow_balance': balance,
                'flow_signal': 1 if balance > 0.3 else (-1 if balance < -0.3 else 0)
            }

        except Exception as e:
            logging.error(f"订单流分析失败: {e}")
            return {}

    def analyze_technical_signals(self, klines_df):
        """分析技术指标信号"""
        print("\n📈 技术指标信号分析:")

        try:
            if len(klines_df) < 20:
                print("   ❌ K线数据不足")
                return {}

            closes = klines_df['close'].values.astype(float)

            # 价格动量
            returns = np.diff(closes) / closes[:-1]
            momentum_5 = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            momentum_10 = np.mean(returns[-10:]) if len(returns) >= 10 else 0

            print(f"   5分钟动量: {momentum_5:.4f}")
            print(f"   10分钟动量: {momentum_10:.4f}")
            print(f"   动量阈值: ±{self.config['momentum_threshold']}")
            print(f"   动量信号: {'✅ 看多' if momentum_5 > self.config['momentum_threshold'] else '✅ 看空' if momentum_5 < -self.config['momentum_threshold'] else '❌ 中性'}")

            # 波动率
            volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0
            print(f"   波动率: {volatility:.4f}")
            print(f"   波动率范围: {self.config['volatility_min']:.2f} - {self.config['volatility_max']:.2f}")
            print(f"   波动率检查: {'✅ 适合' if self.config['volatility_min'] < volatility < self.config['volatility_max'] else '❌ 不适合'}")

            # 移动平均线
            if len(closes) >= 10:
                ma_5 = np.mean(closes[-5:])
                ma_10 = np.mean(closes[-10:])
                current_price = closes[-1]
                price_vs_ma5 = (current_price - ma_5) / ma_5
                price_vs_ma10 = (current_price - ma_10) / ma_10

                print(f"   当前价格: ${current_price:.2f}")
                print(f"   MA5: ${ma_5:.2f} ({price_vs_ma5:.2%})")
                print(f"   MA10: ${ma_10:.2f} ({price_vs_ma10:.2%})")

            return {
                'momentum_5': momentum_5,
                'volatility': volatility,
                'tech_signal': 1 if momentum_5 > self.config['momentum_threshold'] else (-1 if momentum_5 < -self.config['momentum_threshold'] else 0)
            }

        except Exception as e:
            logging.error(f"技术指标分析失败: {e}")
            return {}

    def evaluate_ensemble_signal(self, lob_result, flow_result, tech_result):
        """评估集成信号"""
        print("\n🎯 集成信号评估:")

        signals = {
            '订单簿': lob_result.get('lob_signal', 0),
            '订单流': flow_result.get('flow_signal', 0),
            '技术指标': tech_result.get('tech_signal', 0)
        }

        print("   各组件信号:")
        for component, signal in signals.items():
            signal_text = "买入" if signal == 1 else "卖出" if signal == -1 else "中性"
            print(f"     {component}: {signal_text}")

        # 集成权重
        weights = {'订单簿': 0.4, '订单流': 0.3, '技术指标': 0.3}

        weighted_sum = 0
        total_weight = 0
        valid_signals = 0

        for component, signal in signals.items():
            if signal != 0:
                weight = weights[component]
                weighted_sum += signal * weight
                total_weight += weight
                valid_signals += 1

        if total_weight > 0:
            ensemble_signal = 1 if weighted_sum / total_weight > 0.3 else (-1 if weighted_sum / total_weight < -0.3 else 0)
            signal_strength = abs(weighted_sum / total_weight)
        else:
            ensemble_signal = 0
            signal_strength = 0

        print(f"\n   集成结果:")
        print(f"   有效信号数: {valid_signals}/3")
        print(f"   加权强度: {signal_strength:.2f}")
        print(f"   最终信号: {'买入' if ensemble_signal == 1 else '卖出' if ensemble_signal == -1 else '无信号'}")
        print(f"   信号强度要求: 0.4")
        print(f"   强度检查: {'✅ 达标' if signal_strength >= 0.4 else '❌ 不足'}")

        # 综合评估
        print(f"\n🔍 无信号原因分析:")

        issues = []

        # 检查各组件
        if lob_result.get('lob_signal', 0) == 0:
            issues.append(f"订单簿不平衡不足 ({lob_result.get('imbalance', 0):.3f} < {self.config['imbalance_threshold']})")

        if flow_result.get('flow_signal', 0) == 0:
            issues.append(f"订单流不平衡不足 ({flow_result.get('order_flow_balance', 0):.3f})")

        if tech_result.get('tech_signal', 0) == 0:
            issues.append(f"动量不足 ({tech_result.get('momentum_5', 0):.4f})")

        if signal_strength < 0.4:
            issues.append(f"整体信号强度不足 ({signal_strength:.2f} < 0.4)")

        # 检查其他过滤条件
        if lob_result.get('spread_bps', 0) > self.config['spread_threshold'] * 10000:
            issues.append(f"价差过大 ({lob_result.get('spread_bps', 0):.1f} bps)")

        if not (self.config['volatility_min'] < tech_result.get('volatility', 0) < self.config['volatility_max']):
            issues.append(f"波动率不合适 ({tech_result.get('volatility', 0):.4f})")

        if issues:
            print("   主要问题:")
            for i, issue in enumerate(issues, 1):
                print(f"     {i}. {issue}")
        else:
            print("   ✅ 所有条件都满足，应该有交易信号")

        return ensemble_signal, signal_strength, issues

    def run_diagnosis(self):
        """运行完整诊断"""
        print("🔍 80%胜率系统信号诊断分析")
        print("="*60)

        # 获取市场数据
        lob_data, trades_data, klines_df = self.get_market_data()
        if not lob_data or not trades_data or klines_df is None:
            print("❌ 无法获取市场数据")
            return

        # 分析各组件信号
        lob_result = self.analyze_lob_signals(lob_data)
        flow_result = self.analyze_order_flow_signals(trades_data)
        tech_result = self.analyze_technical_signals(klines_df)

        # 评估集成信号
        ensemble_signal, signal_strength, issues = self.evaluate_ensemble_signal(lob_result, flow_result, tech_result)

        # 提供优化建议
        print(f"\n💡 优化建议:")

        if not issues:
            print("   ✅ 系统运行正常，应该能生成信号")
        else:
            print("   📊 参数调整建议:")

            if "订单簿不平衡不足" in str(issues):
                print("     • 降低订单不平衡阈值 (如 0.25 → 0.2)")
                print("     • 增加订单簿分析层级")

            if "动量不足" in str(issues):
                print("     • 降低动量阈值 (如 0.001 → 0.0005)")
                print("     • 缩短动量计算周期")

            if "信号强度不足" in str(issues):
                print("     • 降低信号强度要求 (如 0.4 → 0.3)")
                print("     • 调整各组件权重分配")

            if "价差过大" in str(issues):
                print("     • 提高价差容忍度")
                print("     • 选择流动性更好的时段")

            print("   📅 市场时机建议:")
            print("     • 选择高波动率时段交易")
            print("     • 避免市场平静期")
            print("     • 关注重要经济数据发布时间")

def main():
    """主函数"""
    diagnosis = SignalDiagnosis()
    diagnosis.run_diagnosis()

if __name__ == "__main__":
    main()