#!/usr/bin/env python3
"""
数据管理和准备系统
为机器学习模型提供高质量的数据
"""

import ccxt
import pandas as pd
import numpy as np
import time
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """数据管理器"""

    def __init__(self, db_path="market_data.db"):
        self.db_path = db_path
        self.exchange = ccxt.binance()
        self.init_database()

    def init_database(self):
        """初始化数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 创建K线数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS klines_5m (
                    timestamp INTEGER PRIMARY KEY,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 创建特征数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    returns REAL,
                    rsi REAL,
                    ma5 REAL,
                    ma10 REAL,
                    ma20 REAL,
                    bb_position REAL,
                    volatility REAL,
                    volume_ratio REAL,
                    momentum REAL,
                    label INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()
            conn.close()
            logger.info("✅ 数据库初始化完成")

        except Exception as e:
            logger.error(f"❌ 数据库初始化失败: {e}")

    def fetch_historical_data(self, symbol='ETH/USDT', timeframe='5m', limit=1000):
        """获取历史数据"""
        try:
            logger.info(f"📥 获取 {symbol} {timeframe} 历史数据，数量: {limit}")

            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            logger.info(f"✅ 获取到 {len(df)} 条数据")
            logger.info(f"📅 时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")

            return df

        except Exception as e:
            logger.error(f"❌ 获取历史数据失败: {e}")
            return None

    def save_to_database(self, df):
        """保存数据到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)

            # 转换时间戳为Unix时间戳
            df_to_save = df.copy()
            df_to_save['timestamp'] = df_to_save['timestamp'].astype(int) // 1000

            # 保存K线数据
            df_to_save[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_sql(
                'klines_5m', conn, if_exists='replace', index=False
            )

            conn.commit()
            conn.close()

            logger.info(f"💾 成功保存 {len(df)} 条数据到数据库")
            return True

        except Exception as e:
            logger.error(f"❌ 保存数据失败: {e}")
            return False

    def calculate_features(self, df):
        """计算技术指标特征"""
        logger.info("🔧 计算技术指标特征...")

        df = df.copy()

        # 基础价格指标
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # 移动平均线
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 布林带
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # 波动率
        df['volatility'] = df['returns'].rolling(10).std()

        # 成交量指标
        df['volume_ma'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # 动量
        df['momentum'] = df['close'].pct_change(5)

        # 价格相对位置
        df['price_position'] = (df['close'] - df['low'].rolling(20).min()) / \
                              (df['high'].rolling(20).max() - df['low'].rolling(20).min())

        logger.info(f"✅ 计算了 {len(df.columns)} 个特征")
        return df

    def create_labels(self, df, horizon=1, threshold=0.001):
        """创建预测标签"""
        logger.info(f"🏷️ 创建预测标签 (horizon={horizon}, threshold={threshold})")

        df = df.copy()

        # 计算未来收益
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1

        # 创建标签
        df['label'] = (df['future_return'] > threshold).astype(int)

        # 移除无法计算标签的最后几行
        df = df.dropna(subset=['future_return', 'label'])

        label_dist = df['label'].value_counts()
        logger.info(f"📊 标签分布: 上涨={label_dist.get(1, 0)}, 下跌/横盘={label_dist.get(0, 0)}")

        return df

    def prepare_ml_dataset(self, symbol='ETH/USDT', limit=1000):
        """准备机器学习数据集"""
        logger.info("🚀 开始准备机器学习数据集")

        # 1. 获取原始数据
        raw_data = self.fetch_historical_data(symbol, '5m', limit)
        if raw_data is None:
            return None

        # 2. 计算特征
        feature_data = self.calculate_features(raw_data)

        # 3. 创建标签
        labeled_data = self.create_labels(feature_data)

        # 4. 选择最终的特征列
        feature_columns = [
            'returns', 'log_returns', 'rsi', 'ma5', 'ma10', 'ma20',
            'bb_position', 'volatility', 'volume_ratio', 'momentum', 'price_position'
        ]

        # 5. 清理数据
        final_data = labeled_data[feature_columns + ['label']].dropna()

        if len(final_data) < 100:
            logger.error(f"❌ 数据不足: {len(final_data)} < 100")
            return None

        logger.info(f"✅ 数据集准备完成: {len(final_data)} 样本, {len(feature_columns)} 特征")

        return final_data

    def save_features_to_db(self, df):
        """保存特征数据到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)

            # 准备数据
            feature_df = df.copy()
            feature_df['timestamp'] = feature_df.index.astype(int) // 1000

            # 选择要保存的列
            save_columns = ['timestamp', 'returns', 'rsi', 'ma5', 'ma10', 'ma20',
                           'bb_position', 'volatility', 'volume_ratio', 'momentum', 'label']

            feature_df[save_columns].to_sql('features', conn, if_exists='replace', index=False)

            conn.commit()
            conn.close()

            logger.info(f"💾 成功保存 {len(feature_df)} 条特征数据")
            return True

        except Exception as e:
            logger.error(f"❌ 保存特征数据失败: {e}")
            return False

    def get_latest_data(self, count=50):
        """获取最新的数据"""
        try:
            conn = sqlite3.connect(self.db_path)

            query = f'''
                SELECT * FROM klines_5m
                ORDER BY timestamp DESC
                LIMIT {count}
            '''

            df = pd.read_sql_query(query, conn)
            conn.close()

            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"❌ 获取最新数据失败: {e}")
            return None

def main():
    """主函数 - 演示数据管理流程"""
    print("📊 数据管理系统演示")
    print("="*50)

    # 创建数据管理器
    dm = DataManager()

    # 准备数据集
    print("\n🚀 步骤1: 准备机器学习数据集")
    dataset = dm.prepare_ml_dataset(limit=500)

    if dataset is not None:
        print(f"✅ 数据集准备成功!")
        print(f"   样本数量: {len(dataset)}")
        print(f"   特征数量: {len(dataset.columns) - 1}")
        print(f"   标签分布: {dataset['label'].value_counts().to_dict()}")

        # 保存到数据库
        print("\n💾 步骤2: 保存数据到数据库")
        success = dm.save_features_to_db(dataset)

        if success:
            print("✅ 数据保存成功!")
        else:
            print("❌ 数据保存失败!")
    else:
        print("❌ 数据集准备失败!")

    # 测试数据获取
    print("\n📥 步骤3: 测试数据获取")
    latest_data = dm.get_latest_data(20)

    if latest_data is not None:
        print(f"✅ 获取最新数据成功: {len(latest_data)} 条记录")
        print(f"   时间范围: {latest_data['timestamp'].min()} 到 {latest_data['timestamp'].max()}")
        print(f"   最新价格: ${latest_data['close'].iloc[-1]:.2f}")
    else:
        print("❌ 获取最新数据失败!")

    print("\n🎉 数据管理演示完成!")

if __name__ == "__main__":
    main()