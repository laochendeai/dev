#!/usr/bin/env python3
"""
高级胜率优化回测系统
基于胜率改进策略的多参数优化框架
"""

import numpy as np
import pandas as pd
import ccxt
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import itertools
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TradingParameters:
    """交易参数配置"""
    # RSI参数
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70

    # 移动平均线参数
    ma_short: int = 12
    ma_long: int = 26

    # MACD参数
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # 布林带参数
    bb_period: int = 20
    bb_std: float = 2.0

    # 信号过滤参数
    min_signal_strength: float = 0.6
    volume_confirmation: bool = True
    volume_spike_threshold: float = 1.5

    # 风险管理参数
    use_atr_stops: bool = True
    atr_period: int = 14
    atr_multiplier: float = 2.0
    risk_per_trade: float = 0.02

    # 入场策略参数
    use_pullback_entry: bool = True
    pullback_level: float = 0.382
    require_breakout_confirmation: bool = True

    # 时间过滤参数
    min_time_between_trades: int = 300  # 5分钟
    max_holding_time: int = 3600        # 1小时

class AdvancedSignalGenerator:
    """高级信号生成器"""

    def __init__(self, params: TradingParameters):
        self.params = params

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.params.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.params.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 移动平均线
        df['ma_short'] = df['close'].rolling(window=self.params.ma_short).mean()
        df['ma_long'] = df['close'].rolling(window=self.params.ma_long).mean()

        # MACD
        exp1 = df['close'].ewm(span=self.params.macd_fast).mean()
        exp2 = df['close'].ewm(span=self.params.macd_slow).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=self.params.macd_signal).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # 布林带
        df['bb_middle'] = df['close'].rolling(window=self.params.bb_period).mean()
        bb_std = df['close'].rolling(window=self.params.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * self.params.bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std * self.params.bb_std)

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = tr.rolling(window=self.params.atr_period).mean()

        # 成交量指标
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # 动量指标
        df['momentum'] = df['close'].pct_change(periods=5)
        df['price_change'] = df['close'].pct_change()

        return df

    def generate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        df = df.copy()

        # 基础信号
        df['rsi_signal'] = 0
        df.loc[df['rsi'] < self.params.rsi_oversold, 'rsi_signal'] = 1
        df.loc[df['rsi'] > self.params.rsi_overbought, 'rsi_signal'] = -1

        df['ma_signal'] = 0
        df.loc[df['ma_short'] > df['ma_long'], 'ma_signal'] = 1
        df.loc[df['ma_short'] < df['ma_long'], 'ma_signal'] = -1

        df['macd_signal'] = 0
        df.loc[(df['macd'] > df['macd_signal']) & (df['macd_histogram'] > 0), 'macd_signal'] = 1
        df.loc[(df['macd'] < df['macd_signal']) & (df['macd_histogram'] < 0), 'macd_signal'] = -1

        df['bb_signal'] = 0
        df.loc[df['close'] < df['bb_lower'], 'bb_signal'] = 1
        df.loc[df['close'] > df['bb_upper'], 'bb_signal'] = -1

        # 成交量确认
        if self.params.volume_confirmation:
            df['volume_confirmed'] = df['volume_ratio'] > self.params.volume_spike_threshold
        else:
            df['volume_confirmed'] = True

        # 综合信号强度计算
        signal_columns = ['rsi_signal', 'ma_signal', 'macd_signal', 'bb_signal']
        df['signal_strength'] = df[signal_columns].sum(axis=1)
        df['signal_strength'] = df['signal_strength'] / len(signal_columns)

        # 生成最终信号
        df['final_signal'] = 0
        buy_condition = (
            (df['signal_strength'] >= self.params.min_signal_strength) &
            (df['volume_confirmed']) &
            (df['momentum'] > 0)  # 动量确认
        )
        sell_condition = (
            (df['signal_strength'] <= -self.params.min_signal_strength) &
            (df['volume_confirmed']) &
            (df['momentum'] < 0)  # 动量确认
        )

        df.loc[buy_condition, 'final_signal'] = 1
        df.loc[sell_condition, 'final_signal'] = -1

        return df

class AdvancedBacktester:
    """高级回测系统"""

    def __init__(self, params: TradingParameters, initial_balance: float = 10000):
        self.params = params
        self.initial_balance = initial_balance
        self.signal_generator = AdvancedSignalGenerator(params)

    def fetch_data(self, symbol: str = 'ETH/USDT', timeframe: str = '5m', limit: int = 1000) -> pd.DataFrame:
        """获取交易数据"""
        try:
            exchange = ccxt.binance()
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            return df
        except Exception as e:
            print(f"获取数据失败: {e}")
            return pd.DataFrame()

    def calculate_dynamic_stops(self, entry_price: float, atr: float, direction: str) -> Tuple[float, float]:
        """计算动态止损止盈"""
        if not self.params.use_atr_stops:
            if direction == 'long':
                stop_loss = entry_price * 0.98  # 2% 止损
                take_profit = entry_price * 1.04  # 4% 止盈
            else:
                stop_loss = entry_price * 1.02
                take_profit = entry_price * 0.96
        else:
            stop_distance = atr * self.params.atr_multiplier
            if direction == 'long':
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + (stop_distance * 2)  # 1:2 盈亏比
            else:
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - (stop_distance * 2)

        return stop_loss, take_profit

    def backtest(self, data: pd.DataFrame) -> Dict:
        """执行回测"""
        if data.empty:
            return {'error': '数据为空'}

        # 计算指标
        data = self.signal_generator.calculate_indicators(data)
        data = self.signal_generator.generate_trading_signals(data)

        balance = self.initial_balance
        position = 0
        trades = []
        equity_curve = [self.initial_balance]

        last_trade_time = None
        position_entry_time = None
        entry_price = 0
        stop_loss = 0
        take_profit = 0

        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_price = row['close']

            # 时间过滤
            if last_trade_time and (timestamp - last_trade_time).total_seconds() < self.params.min_time_between_trades:
                continue

            # 持仓时间检查
            if position != 0 and position_entry_time:
                holding_time = (timestamp - position_entry_time).total_seconds()
                if holding_time > self.params.max_holding_time:
                    # 时间止损
                    if position > 0:
                        pnl = (current_price - entry_price) * position
                        balance += position
                    else:
                        pnl = (entry_price - current_price) * abs(position)
                        balance += abs(position)

                    trades.append({
                        'entry_time': position_entry_time,
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'quantity': abs(position),
                        'pnl': pnl,
                        'balance': balance,
                        'exit_reason': 'time_stop'
                    })

                    position = 0
                    position_entry_time = None
                    last_trade_time = timestamp

            # 检查止损止盈
            if position != 0:
                if position > 0:
                    if current_price <= stop_loss or current_price >= take_profit:
                        pnl = (current_price - entry_price) * position
                        balance += position

                        exit_reason = 'stop_loss' if current_price <= stop_loss else 'take_profit'
                        trades.append({
                            'entry_time': position_entry_time,
                            'exit_time': timestamp,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'quantity': position,
                            'pnl': pnl,
                            'balance': balance,
                            'exit_reason': exit_reason
                        })

                        position = 0
                        position_entry_time = None
                        last_trade_time = timestamp

                elif position < 0:
                    if current_price >= stop_loss or current_price <= take_profit:
                        pnl = (entry_price - current_price) * abs(position)
                        balance += abs(position)

                        exit_reason = 'stop_loss' if current_price >= stop_loss else 'take_profit'
                        trades.append({
                            'entry_time': position_entry_time,
                            'exit_time': timestamp,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'quantity': abs(position),
                            'pnl': pnl,
                            'balance': balance,
                            'exit_reason': exit_reason
                        })

                        position = 0
                        position_entry_time = None
                        last_trade_time = timestamp

            # 信号处理
            if position == 0 and row['final_signal'] != 0:
                signal = row['final_signal']
                atr = row['atr']

                # 计算仓位大小
                risk_amount = balance * self.params.risk_per_trade
                if self.params.use_atr_stops and atr > 0:
                    stop_distance = atr * self.params.atr_multiplier
                    quantity = risk_amount / stop_distance
                else:
                    quantity = risk_amount / (entry_price * 0.02)  # 2%风险

                quantity = min(quantity, balance * 0.1)  # 最大10%仓位

                if signal > 0:
                    # 做多
                    position = quantity
                    entry_price = current_price
                    position_entry_time = timestamp
                    stop_loss, take_profit = self.calculate_dynamic_stops(entry_price, atr, 'long')
                    balance -= quantity  # 扣除保证金

                elif signal < 0:
                    # 做空
                    position = -quantity
                    entry_price = current_price
                    position_entry_time = timestamp
                    stop_loss, take_profit = self.calculate_dynamic_stops(entry_price, atr, 'short')
                    balance -= quantity  # 扣除保证金

            equity_curve.append(balance + (position * current_price if position != 0 else 0))

        # 平仓剩余持仓
        if position != 0:
            current_price = data.iloc[-1]['close']
            if position > 0:
                pnl = (current_price - entry_price) * position
                balance += position
            else:
                pnl = (entry_price - current_price) * abs(position)
                balance += abs(position)

            trades.append({
                'entry_time': position_entry_time,
                'exit_time': data.index[-1],
                'entry_price': entry_price,
                'exit_price': current_price,
                'quantity': abs(position),
                'pnl': pnl,
                'balance': balance,
                'exit_reason': 'end_of_test'
            })

        # 计算统计指标
        if trades:
            total_trades = len(trades)
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]

            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            total_pnl = sum(t['pnl'] for t in trades)
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

            equity_series = pd.Series(equity_curve)
            max_drawdown = (equity_series.cummax() - equity_series).max()
            max_drawdown_pct = max_drawdown / equity_series.max() if equity_series.max() > 0 else 0

            profit_factor = abs(sum(t['pnl'] for t in winning_trades)) / abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf')

            sharpe_ratio = (np.mean(equity_series.pct_change()) * 252) / (np.std(equity_series.pct_change()) * np.sqrt(252)) if len(equity_series) > 1 else 0

        else:
            win_rate = 0
            total_trades = 0
            total_pnl = 0
            avg_pnl = 0
            max_drawdown = 0
            max_drawdown_pct = 0
            profit_factor = 0
            sharpe_ratio = 0

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': balance,
            'return_pct': (balance - self.initial_balance) / self.initial_balance,
            'trades': trades,
            'equity_curve': equity_curve
        }

class ParameterOptimizer:
    """参数优化器"""

    def __init__(self):
        self.optimization_results = []

    def generate_parameter_combinations(self) -> List[TradingParameters]:
        """生成参数组合"""
        # 基于胜率改进策略的参数范围
        rsi_periods = [7, 14, 21]
        rsi_oversold_levels = [25, 30, 35]
        rsi_overbought_levels = [70, 75, 80]

        ma_short_periods = [5, 8, 12]
        ma_long_periods = [21, 26, 34]

        bb_periods = [15, 20, 25]
        bb_stds = [1.8, 2.0, 2.2]

        signal_strengths = [0.4, 0.6, 0.8]
        volume_thresholds = [1.2, 1.5, 2.0]

        atr_multipliers = [1.5, 2.0, 2.5]
        risk_per_trades = [0.01, 0.02, 0.03]

        # 生成组合（为了避免组合爆炸，选择部分关键参数进行优化）
        combinations = []

        # 高胜率组合
        for rsi_period, rsi_os, rsi_ob, ma_s, ma_l, bb_p, bb_std, sig_str, vol_thr, atr_mult, risk in itertools.product(
            rsi_periods, rsi_oversold_levels, rsi_overbought_levels, ma_short_periods, ma_long_periods,
            bb_periods, bb_stds, signal_strengths, volume_thresholds, atr_multipliers, risk_per_trades
        ):
            params = TradingParameters(
                rsi_period=rsi_period,
                rsi_oversold=rsi_os,
                rsi_overbought=rsi_ob,
                ma_short=ma_s,
                ma_long=ma_l,
                bb_period=bb_p,
                bb_std=bb_std,
                min_signal_strength=sig_str,
                volume_spike_threshold=vol_thr,
                atr_multiplier=atr_mult,
                risk_per_trade=risk
            )
            combinations.append(params)

        # 添加一些预定义的优秀组合
        optimized_combinations = [
            TradingParameters(rsi_period=7, rsi_oversold=25, rsi_overbought=80, ma_short=5, ma_long=21,
                            bb_period=15, bb_std=1.8, min_signal_strength=0.8, volume_spike_threshold=2.0,
                            atr_multiplier=1.5, risk_per_trade=0.01),
            TradingParameters(rsi_period=14, rsi_oversold=30, rsi_overbought=75, ma_short=8, ma_long=26,
                            bb_period=20, bb_std=2.0, min_signal_strength=0.6, volume_spike_threshold=1.5,
                            atr_multiplier=2.0, risk_per_trade=0.02),
            TradingParameters(rsi_period=21, rsi_oversold=35, rsi_overbought=70, ma_short=12, ma_long=34,
                            bb_period=25, bb_std=2.2, min_signal_strength=0.4, volume_spike_threshold=1.2,
                            atr_multiplier=2.5, risk_per_trade=0.03)
        ]

        combinations.extend(optimized_combinations)

        return combinations[:50]  # 限制组合数量以控制运行时间

    def optimize_parameters(self, data: pd.DataFrame) -> List[Dict]:
        """执行参数优化"""
        print("🚀 开始参数优化...")

        combinations = self.generate_parameter_combinations()
        print(f"📊 总共测试 {len(combinations)} 种参数组合")

        results = []

        for i, params in enumerate(combinations):
            if (i + 1) % 10 == 0:
                print(f"进度: {i + 1}/{len(combinations)} ({((i + 1)/len(combinations)*100):.1f}%)")

            try:
                backtester = AdvancedBacktester(params)
                result = backtester.backtest(data)

                # 计算综合评分
                score = self.calculate_score(result)

                result_data = {
                    'params': params,
                    'metrics': result,
                    'score': score
                }

                results.append(result_data)

            except Exception as e:
                print(f"参数组合 {i+1} 测试失败: {e}")
                continue

        # 按评分排序
        results.sort(key=lambda x: x['score'], reverse=True)

        self.optimization_results = results
        return results

    def calculate_score(self, metrics: Dict) -> float:
        """计算综合评分"""
        if metrics['total_trades'] < 5:
            return 0

        # 权重设置（基于胜率改进策略的重要性）
        win_rate_weight = 0.3
        profit_factor_weight = 0.25
        return_weight = 0.2
        sharpe_weight = 0.15
        drawdown_weight = 0.1

        # 标准化指标
        win_rate_score = metrics['win_rate']
        profit_factor_score = min(metrics['profit_factor'] / 3, 1)  # 3以上为满分
        return_score = min(max(metrics['return_pct'], 0), 0.5)  # 最高50%
        sharpe_score = min(max(metrics['sharpe_ratio'], 0), 2) / 2  # 最高2
        drawdown_score = 1 - min(metrics['max_drawdown_pct'], 0.2)  # 20%以下

        total_score = (
            win_rate_score * win_rate_weight +
            profit_factor_score * profit_factor_weight +
            return_score * return_weight +
            sharpe_score * sharpe_weight +
            drawdown_score * drawdown_weight
        )

        return total_score

    def save_results(self, filename: str = 'advanced_optimization_results.json'):
        """保存优化结果"""
        if not self.optimization_results:
            print("没有优化结果可保存")
            return

        # 准备保存的数据
        save_data = {
            'optimization_date': datetime.now().isoformat(),
            'total_combinations_tested': len(self.optimization_results),
            'top_results': []
        }

        for i, result in enumerate(self.optimization_results[:20]):  # 保存前20个结果
            params = result['params']
            metrics = result['metrics']

            save_data['top_results'].append({
                'rank': i + 1,
                'score': result['score'],
                'parameters': {
                    'rsi_period': params.rsi_period,
                    'rsi_oversold': params.rsi_oversold,
                    'rsi_overbought': params.rsi_overbought,
                    'ma_short': params.ma_short,
                    'ma_long': params.ma_long,
                    'bb_period': params.bb_period,
                    'bb_std': params.bb_std,
                    'min_signal_strength': params.min_signal_strength,
                    'volume_spike_threshold': params.volume_spike_threshold,
                    'atr_multiplier': params.atr_multiplier,
                    'risk_per_trade': params.risk_per_trade
                },
                'performance': {
                    'win_rate': metrics['win_rate'],
                    'total_trades': metrics['total_trades'],
                    'return_pct': metrics['return_pct'],
                    'profit_factor': metrics['profit_factor'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown_pct': metrics['max_drawdown_pct']
                }
            })

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"✅ 优化结果已保存到 {filename}")

def main():
    """主函数"""
    print("🎯 高级胜率优化回测系统")
    print("基于胜率改进策略的多参数优化")
    print("="*60)

    # 获取数据
    print("📈 获取市场数据...")
    optimizer = ParameterOptimizer()
    backtester = AdvancedBacktester(TradingParameters())

    data = backtester.fetch_data('ETH/USDT', '5m', 2000)
    if data.empty:
        print("❌ 无法获取数据，退出")
        return

    print(f"✅ 获取到 {len(data)} 条数据点")
    print(f"数据时间范围: {data.index[0]} 到 {data.index[-1]}")

    # 执行优化
    results = optimizer.optimize_parameters(data)

    if not results:
        print("❌ 优化失败，没有有效结果")
        return

    print(f"\n🏆 优化完成！共测试了 {len(results)} 种参数组合")

    # 显示前10个结果
    print(f"\n📊 Top 10 优化结果:")
    print("-"*80)
    print(f"{'排名':<4} {'评分':<8} {'胜率':<8} {'交易数':<8} {'收益率':<10} {'盈亏比':<8} {'回撤':<8}")
    print("-"*80)

    for i, result in enumerate(results[:10]):
        metrics = result['metrics']
        print(f"{i+1:<4} {result['score']:.3f}    {metrics['win_rate']:.1%}    {metrics['total_trades']:<8} "
              f"{metrics['return_pct']:.1%}    {metrics['profit_factor']:.2f}    {metrics['max_drawdown_pct']:.1%}")

    # 最佳参数分析
    best_result = results[0]
    best_params = best_result['params']
    best_metrics = best_result['metrics']

    print(f"\n🎯 最佳参数组合:")
    print(f"   RSI: 周期={best_params.rsi_period}, 超卖={best_params.rsi_oversold}, 超买={best_params.rsi_overbought}")
    print(f"   MA: 短期={best_params.ma_short}, 长期={best_params.ma_long}")
    print(f"   布林带: 周期={best_params.bb_period}, 标准差={best_params.bb_std}")
    print(f"   信号强度: {best_params.min_signal_strength}")
    print(f"   成交量确认: {best_params.volume_spike_threshold}")
    print(f"   ATR倍数: {best_params.atr_multiplier}")
    print(f"   风险比例: {best_params.risk_per_trade:.1%}")

    print(f"\n📈 最佳性能指标:")
    print(f"   胜率: {best_metrics['win_rate']:.1%}")
    print(f"   总交易数: {best_metrics['total_trades']}")
    print(f"   总收益率: {best_metrics['return_pct']:.1%}")
    print(f"   平均盈亏: ${best_metrics['avg_pnl']:.2f}")
    print(f"   盈亏比: {best_metrics['profit_factor']:.2f}")
    print(f"   最大回撤: {best_metrics['max_drawdown_pct']:.1%}")
    print(f"   夏普比率: {best_metrics['sharpe_ratio']:.2f}")

    # 保存结果
    optimizer.save_results()

    # 预期改进分析
    current_winrate = 0.0  # 当前系统胜率
    optimized_winrate = best_metrics['win_rate']
    improvement = optimized_winrate - current_winrate

    print(f"\n🚀 预期胜率改进:")
    print(f"   当前胜率: {current_winrate:.1%}")
    print(f"   优化胜率: {optimized_winrate:.1%}")
    print(f"   改进幅度: {improvement:.1%} ({improvement/current_winrate*100 if current_winrate > 0 else 0:.1f}%)")

    print(f"\n✅ 下一步计划:")
    print(f"   1. 使用最佳参数创建实时交易系统")
    print(f"   2. 进行小规模实盘测试")
    print(f"   3. 持续监控和调整参数")
    print(f"   4. 扩展到更多交易对")

if __name__ == "__main__":
    main()