#!/usr/bin/env python3
"""
ETH历史数据获取器
使用CCXT库从多个交易所获取ETH历史数据
"""

import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ETHDataFetcher:
    def __init__(self):
        # 支持的交易所列表
        self.exchanges = {
            'binance': ccxt.binance(),
            'coinbase': ccxt.coinbase(),
            'kraken': ccxt.kraken(),
            'kucoin': ccxt.kucoin(),
        }

    def fetch_ohlcv_data(self, exchange_name='binance', symbol='ETH/USDT', timeframe='1h', days=30):
        """获取OHLCV数据"""
        try:
            exchange = self.exchanges[exchange_name]

            # 计算开始时间
            since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())

            # 获取数据
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)

            # 转换为DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            logger.info(f"成功从 {exchange_name} 获取 {len(df)} 条数据")
            return df

        except Exception as e:
            logger.error(f"从 {exchange_name} 获取数据失败: {e}")
            return None

    def fetch_multiple_exchanges(self, symbol='ETH/USDT', timeframe='1h', days=30):
        """从多个交易所获取数据并合并"""
        all_data = {}

        for exchange_name in self.exchanges.keys():
            logger.info(f"正在从 {exchange_name} 获取数据...")
            data = self.fetch_ohlcv_data(exchange_name, symbol, timeframe, days)

            if data is not None:
                all_data[exchange_name] = data
                time.sleep(1)  # 避免API限制

        return all_data

    def save_to_csv(self, data, filename):
        """保存数据到CSV文件"""
        if isinstance(data, dict):
            # 多个交易所的数据
            with open(filename, 'w') as f:
                f.write("exchange,timestamp,open,high,low,close,volume\n")

                for exchange_name, df in data.items():
                    for timestamp, row in df.iterrows():
                        f.write(f"{exchange_name},{timestamp},{row['open']},{row['high']},{row['low']},{row['close']},{row['volume']}\n")
        else:
            # 单个交易所的数据
            data.to_csv(filename)

        logger.info(f"数据已保存到 {filename}")

def main():
    """主函数"""
    fetcher = ETHDataFetcher()

    # 获取最近30天的1小时数据
    logger.info("开始获取ETH历史数据...")

    # 从Binance获取数据
    binance_data = fetcher.fetch_ohlcv_data('binance', 'ETH/USDT', '1h', days=30)
    if binance_data is not None:
        fetcher.save_to_csv(binance_data, 'eth_binance_1h_30d.csv')

    # 从多个交易所获取数据
    all_data = fetcher.fetch_multiple_exchanges('ETH/USDT', '1h', days=7)  # 减少天数以避免过多请求
    if all_data:
        fetcher.save_to_csv(all_data, 'eth_multiple_exchanges_1h_7d.csv')

    logger.info("数据获取完成!")

if __name__ == "__main__":
    main()