#!/usr/bin/env python3
"""
ETH剥头皮交易策略回测框架
使用历史数据进行参数优化和胜率分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import itertools
import logging
from typing import Dict, List, Tuple, Any
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ETHBacktester:
    def __init__(self, data_file: str):
        """初始化回测器"""
        self.data = self.load_data(data_file)
        self.results = []

    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载历史数据"""
        try:
            if 'multiple_exchanges' in file_path:
                # 多交易所数据格式
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df[df['exchange'] == 'binance']  # 使用Binance数据
                df.set_index('timestamp', inplace=True)
                df = df[['open', 'high', 'low', 'close', 'volume']]
            else:
                # 单交易所数据格式
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            logger.info(f"成功加载 {len(df)} 条历史数据")
            return df

        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return pd.DataFrame()

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()

        # 移动平均线
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()

        # RSI
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        df['rsi_14'] = calculate_rsi(df['close'])

        # 布林带
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        # 价格变动
        df['price_change_1h'] = df['close'].pct_change()
        df['price_change_6h'] = df['close'].pct_change(6)

        return df

    def generate_signals(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """生成交易信号"""
        df = df.copy()

        # 初始化信号列
        df['signal'] = 0  # 0: hold, 1: buy, -1: sell

        # 基于参数的信号生成逻辑
        confidence_score = 0

        # 移动平均线信号
        if params['use_ma_signals']:
            ma_condition_buy = (df['ma_5'] > df['ma_20']) & (df['close'] > df['ma_5'])
            ma_condition_sell = (df['ma_5'] < df['ma_20']) & (df['close'] < df['ma_5'])

            df.loc[ma_condition_buy, 'signal'] = 1
            df.loc[ma_condition_sell, 'signal'] = -1

        # RSI信号
        if params['use_rsi_signals']:
            rsi_oversold = df['rsi_14'] < params['rsi_oversold']
            rsi_overbought = df['rsi_14'] > params['rsi_overbought']

            df.loc[rsi_oversold, 'signal'] = 1
            df.loc[rsi_overbought, 'signal'] = -1

        # 布林带信号
        if params['use_bb_signals']:
            bb_buy = df['close'] < df['bb_lower']
            bb_sell = df['close'] > df['bb_upper']

            df.loc[bb_buy, 'signal'] = 1
            df.loc[bb_sell, 'signal'] = -1

        # MACD信号
        if params['use_macd_signals']:
            macd_buy = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
            macd_sell = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))

            df.loc[macd_buy, 'signal'] = 1
            df.loc[macd_sell, 'signal'] = -1

        # 动量信号
        if params['use_momentum_signals']:
            momentum_buy = df['price_change_1h'] > params['momentum_threshold']
            momentum_sell = df['price_change_1h'] < -params['momentum_threshold']

            df.loc[momentum_buy, 'signal'] = 1
            df.loc[momentum_sell, 'signal'] = -1

        return df

    def backtest_strategy(self, params: Dict) -> Dict:
        """执行回测策略"""
        try:
            # 计算技术指标
            df_with_indicators = self.calculate_technical_indicators(self.data)

            # 生成信号
            df_with_signals = self.generate_signals(df_with_indicators, params)

            # 初始化回测变量
            balance = params['initial_balance']
            position = None
            trades = []
            equity_curve = [balance]
            max_balance = balance
            max_drawdown = 0

            for i in range(len(df_with_signals)):
                current_row = df_with_signals.iloc[i]
                current_price = current_row['close']
                signal = current_row['signal']

                # 平仓逻辑
                if position is not None:
                    entry_price = position['entry_price']
                    position_type = position['type']
                    position_size = position['size']

                    if position_type == 'long':
                        pnl_pct = (current_price - entry_price) / entry_price
                    else:  # short
                        pnl_pct = (entry_price - current_price) / entry_price

                    # 检查止盈止损
                    should_close = False
                    close_reason = ""

                    if position_type == 'long':
                        if current_price <= position['stop_loss']:
                            should_close = True
                            close_reason = "止损"
                        elif current_price >= position['take_profit']:
                            should_close = True
                            close_reason = "止盈"
                    else:  # short
                        if current_price >= position['stop_loss']:
                            should_close = True
                            close_reason = "止损"
                        elif current_price <= position['take_profit']:
                            should_close = True
                            close_reason = "止盈"

                    # 强制平仓条件（反向信号）
                    if (position_type == 'long' and signal == -1) or (position_type == 'short' and signal == 1):
                        should_close = True
                        close_reason = "反向信号"

                    if should_close:
                        pnl = position_size * pnl_pct
                        balance += pnl

                        trade = {
                            'entry_time': position['entry_time'],
                            'exit_time': current_row.name,
                            'type': position_type,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'size': position_size,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'reason': close_reason,
                            'holding_hours': (current_row.name - position['entry_time']).total_seconds() / 3600
                        }
                        trades.append(trade)
                        position = None

                # 开仓逻辑
                if position is None and signal != 0:
                    position_type = 'long' if signal == 1 else 'short'
                    position_size = min(params['max_position_size'], balance * params['position_size_ratio'])

                    if position_size > 0:
                        position = {
                            'type': position_type,
                            'entry_price': current_price,
                            'size': position_size,
                            'entry_time': current_row.name,
                            'stop_loss': current_price * (1 - params['stop_loss_pct']) if position_type == 'long' else current_price * (1 + params['stop_loss_pct']),
                            'take_profit': current_price * (1 + params['take_profit_pct']) if position_type == 'long' else current_price * (1 - params['take_profit_pct'])
                        }

                # 更新权益曲线
                if position is not None:
                    unrealized_pnl = 0
                    if position['type'] == 'long':
                        unrealized_pnl = position['size'] * (current_price - position['entry_price']) / position['entry_price']
                    else:
                        unrealized_pnl = position['size'] * (position['entry_price'] - current_price) / position['entry_price']
                    current_equity = balance + unrealized_pnl
                else:
                    current_equity = balance

                equity_curve.append(current_equity)

                # 计算最大回撤
                if current_equity > max_balance:
                    max_balance = current_equity
                drawdown = (max_balance - current_equity) / max_balance
                max_drawdown = max(max_drawdown, drawdown)

            # 计算回测结果
            if not trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'total_return': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0,
                    'avg_trade': 0,
                    'profit_factor': 0
                }

            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]

            total_return = (balance - params['initial_balance']) / params['initial_balance']

            # 计算夏普比率
            returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(24 * 365) if np.std(returns) != 0 else 0

            total_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
            total_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 1
            profit_factor = total_profit / total_loss if total_loss > 0 else 0

            result = {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trades) if trades else 0,
                'total_return': total_return,
                'final_balance': balance,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'avg_trade': np.mean([t['pnl'] for t in trades]) if trades else 0,
                'avg_winning_trade': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
                'avg_losing_trade': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
                'profit_factor': profit_factor,
                'avg_holding_hours': np.mean([t['holding_hours'] for t in trades]) if trades else 0,
                'equity_curve': equity_curve,
                'trades': trades,
                'params': params
            }

            return result

        except Exception as e:
            logger.error(f"回测执行失败: {e}")
            return {'error': str(e)}

    def optimize_parameters(self, param_grid: Dict) -> List[Dict]:
        """参数优化"""
        logger.info("开始参数优化...")

        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        results = []

        for i, combination in enumerate(param_combinations):
            params = dict(zip(param_names, combination))

            # 设置默认参数
            default_params = {
                'initial_balance': 10000,
                'max_position_size': 1000,
                'position_size_ratio': 0.1,
                'use_ma_signals': True,
                'use_rsi_signals': True,
                'use_bb_signals': True,
                'use_macd_signals': True,
                'use_momentum_signals': True,
            }
            params.update(default_params)

            logger.info(f"测试参数组合 {i+1}/{len(param_combinations)}: {params}")

            result = self.backtest_strategy(params)

            if 'error' not in result:
                results.append(result)

        # 按胜率和收益率排序
        results.sort(key=lambda x: (x['win_rate'], x['total_return']), reverse=True)

        logger.info(f"参数优化完成，测试了 {len(results)} 个参数组合")
        return results

    def plot_results(self, results: List[Dict], top_n: int = 5):
        """绘制回测结果"""
        if not results:
            logger.warning("没有结果可以绘制")
            return

        # 选择前N个最佳结果
        top_results = results[:top_n]

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ETH剥头皮策略回测结果', fontsize=16)

        # 权益曲线
        ax1 = axes[0, 0]
        for i, result in enumerate(top_results):
            equity_curve = result['equity_curve']
            ax1.plot(equity_curve, label=f"策略{i+1} (胜率:{result['win_rate']:.2f})")
        ax1.set_title('权益曲线')
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('权益 ($)')
        ax1.legend()
        ax1.grid(True)

        # 胜率对比
        ax2 = axes[0, 1]
        win_rates = [r['win_rate'] * 100 for r in top_results]
        labels = [f"策略{i+1}" for i in range(len(top_results))]
        ax2.bar(labels, win_rates)
        ax2.set_title('胜率对比 (%)')
        ax2.set_ylabel('胜率 (%)')
        ax2.grid(True)

        # 总收益率对比
        ax3 = axes[1, 0]
        total_returns = [r['total_return'] * 100 for r in top_results]
        ax3.bar(labels, total_returns)
        ax3.set_title('总收益率对比 (%)')
        ax3.set_ylabel('收益率 (%)')
        ax3.grid(True)

        # 夏普比率对比
        ax4 = axes[1, 1]
        sharpe_ratios = [r['sharpe_ratio'] for r in top_results]
        ax4.bar(labels, sharpe_ratios)
        ax4.set_title('夏普比率对比')
        ax4.set_ylabel('夏普比率')
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig('eth_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_results(self, results: List[Dict], filename: str = 'eth_backtest_results.json'):
        """保存回测结果"""
        # 准备可序列化的数据
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            if 'equity_curve' in serializable_result:
                serializable_result['equity_curve'] = [float(x) for x in serializable_result['equity_curve']]
            if 'trades' in serializable_result:
                # 转换datetime为字符串
                for trade in serializable_result['trades']:
                    if 'entry_time' in trade:
                        trade['entry_time'] = str(trade['entry_time'])
                    if 'exit_time' in trade:
                        trade['exit_time'] = str(trade['exit_time'])
            serializable_results.append(serializable_result)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f"回测结果已保存到 {filename}")

def main():
    """主函数"""
    # 创建回测器
    backtester = ETHBacktester('eth_binance_1h_30d.csv')

    # 定义参数网格
    param_grid = {
        'take_profit_pct': [0.002, 0.003, 0.005, 0.008, 0.01],  # 0.2% - 1%
        'stop_loss_pct': [0.003, 0.005, 0.008, 0.01, 0.015],     # 0.3% - 1.5%
        'rsi_oversold': [25, 30, 35],
        'rsi_overbought': [65, 70, 75],
        'momentum_threshold': [0.005, 0.008, 0.01, 0.015],      # 0.5% - 1.5%
    }

    # 执行参数优化
    results = backtester.optimize_parameters(param_grid)

    if results:
        # 显示最佳结果
        logger.info("=== 最佳5个参数组合 ===")
        for i, result in enumerate(results[:5]):
            logger.info(f"\n策略 #{i+1}:")
            logger.info(f"  参数: {result['params']}")
            logger.info(f"  总交易次数: {result['total_trades']}")
            logger.info(f"  胜率: {result['win_rate']:.2%}")
            logger.info(f"  总收益率: {result['total_return']:.2%}")
            logger.info(f"  最大回撤: {result['max_drawdown']:.2%}")
            logger.info(f"  夏普比率: {result['sharpe_ratio']:.2f}")
            logger.info(f"  平均交易: ${result['avg_trade']:.2f}")
            logger.info(f"  平均持仓时间: {result['avg_holding_hours']:.1f} 小时")

        # 绘制结果
        backtester.plot_results(results)

        # 保存结果
        backtester.save_results(results)

    else:
        logger.error("没有成功的回测结果")

if __name__ == "__main__":
    main()