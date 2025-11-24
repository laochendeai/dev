#!/usr/bin/env python3
"""
真正可用的机器学习价格预测引擎
基于实际数据和训练的可运行版本
"""

import numpy as np
import pandas as pd
import ccxt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import threading
import json
from datetime import datetime, timedelta
from collections import deque
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('working_ml_predictor.log'),
        logging.StreamHandler()
    ]
)

class WorkingMLPredictor:
    """真正可用的机器学习价格预测器"""

    def __init__(self):
        # 配置
        self.config = {
            'sequence_length': 20,        # 20个5分钟K线 = 100分钟历史
            'prediction_horizon': 1,      # 预测未来1个5分钟
            'min_training_samples': 500,   # 最少训练样本数
            'model_retrain_interval': 100, # 每100次预测重新训练
            'confidence_threshold': 0.6    # 置信度阈值
        }

        # 数据存储
        self.price_history = deque(maxlen=1000)
        self.volume_history = deque(maxlen=1000)
        self.features = deque(maxlen=1000)
        self.labels = deque(maxlen=1000)

        # 模型
        self.scaler = StandardScaler()
        self.rf_model = RandomForestClassifier(
            n_estimators=50,  # 减少数量以加快训练速度
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )

        # 状态
        self.is_trained = False
        self.prediction_count = 0
        self.correct_predictions = 0
        self.total_predictions = 0
        self.model_performance = {
            'rf_accuracy': 0.0,
            'gb_accuracy': 0.0,
            'ensemble_accuracy': 0.0
        }

        # 交易所连接
        self.exchange = ccxt.binance()

        logging.info("🤖 真实机器学习预测器已启动")
        logging.info(f"📊 配置: 序列长度={self.config['sequence_length']}, 预测步长={self.config['prediction_horizon']}")

    def fetch_market_data(self, limit=500):
        """获取历史市场数据用于训练"""
        try:
            logging.info(f"📥 获取 {limit} 条5分钟K线数据...")
            ohlcv = self.exchange.fetch_ohlcv('ETH/USDT', '5m', limit=limit)

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            logging.info(f"✅ 获取到 {len(df)} 条数据，时间范围: {df.index[0]} 到 {df.index[-1]}")
            return df

        except Exception as e:
            logging.error(f"获取市场数据失败: {e}")
            return None

    def calculate_technical_features(self, df):
        """计算技术指标特征"""
        df = df.copy()

        # 价格变化
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # 移动平均线
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()

        # 价格相对移动平均线
        df['price_vs_ma5'] = (df['close'] - df['ma_5']) / df['ma_5']
        df['price_vs_ma10'] = (df['close'] - df['ma_10']) / df['ma_10']
        df['price_vs_ma20'] = (df['close'] - df['ma_20']) / df['ma_20']

        # 移动平均线关系
        df['ma5_vs_ma10'] = (df['ma_5'] - df['ma_10']) / df['ma_10']
        df['ma10_vs_ma20'] = (df['ma_10'] - df['ma_20']) / df['ma_20']

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 布林带
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # 价格波动率
        df['volatility'] = df['returns'].rolling(window=10).std() * np.sqrt(252)  # 年化波动率
        df['price_range'] = (df['high'] - df['low']) / df['close']

        # 成交量指标
        df['volume_ma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['price_volume'] = df['close'] * df['volume']

        # 动量指标
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)

        # 趋势强度
        df['trend_strength'] = abs(df['ma5_vs_ma10'])

        # 支撑阻力水平
        df['resistance'] = df['high'].rolling(window=20).max()
        df['support'] = df['low'].rolling(window=20).min()
        df['price_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])

        return df

    def create_labels(self, df, horizon=1):
        """创建预测标签"""
        df = df.copy()

        # 计算未来收益
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1

        # 定义标签：1=上涨，0=下跌或横盘
        threshold = 0.001  # 0.1%的阈值
        df['label'] = (df['future_return'] > threshold).astype(int)

        return df

    def prepare_training_data(self, df):
        """准备训练数据"""
        # 计算特征
        df_features = self.calculate_technical_features(df)

        # 创建标签
        df_labeled = self.create_labels(df_features)

        # 选择特征列
        feature_columns = [
            'returns', 'log_returns',
            'price_vs_ma5', 'price_vs_ma10', 'price_vs_ma20',
            'ma5_vs_ma10', 'ma10_vs_ma20',
            'rsi', 'bb_position', 'volatility', 'price_range',
            'volume_ratio', 'price_volume',
            'momentum_5', 'momentum_10', 'trend_strength', 'price_position'
        ]

        # 移除包含NaN的行
        df_clean = df_labeled[feature_columns + ['label']].dropna()

        if len(df_clean) < self.config['min_training_samples']:
            logging.warning(f"⚠️ 数据不足: {len(df_clean)} < {self.config['min_training_samples']}")
            return None, None

        X = df_clean[feature_columns].values
        y = df_clean['label'].values

        logging.info(f"📊 训练数据准备完成: {len(X)} 样本, {len(feature_columns)} 特征")
        logging.info(f"📈 标签分布: 上涨={sum(y)}, 下跌={len(y)-sum(y)}")

        return X, y

    def train_models(self, X, y):
        """训练模型"""
        try:
            if X is None or len(X) == 0:
                logging.error("❌ 训练数据为空")
                return False

            # 分割训练和验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 标准化特征
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # 训练随机森林
            logging.info("🌲 训练随机森林模型...")
            start_time = time.time()
            self.rf_model.fit(X_train_scaled, y_train)
            rf_train_time = time.time() - start_time

            # 训练梯度提升
            logging.info("🚀 训练梯度提升模型...")
            start_time = time.time()
            self.gb_model.fit(X_train_scaled, y_train)
            gb_train_time = time.time() - start_time

            # 评估模型
            rf_pred = self.rf_model.predict(X_val_scaled)
            gb_pred = self.gb_model.predict(X_val_scaled)

            self.model_performance['rf_accuracy'] = accuracy_score(y_val, rf_pred)
            self.model_performance['gb_accuracy'] = accuracy_score(y_val, gb_pred)

            # 集成预测（简单平均）
            ensemble_pred = (rf_pred + gb_pred) >= 1
            self.model_performance['ensemble_accuracy'] = accuracy_score(y_val, ensemble_pred)

            self.is_trained = True

            logging.info("✅ 模型训练完成!")
            logging.info(f"   随机森林: {self.model_performance['rf_accuracy']:.3f} (训练时间: {rf_train_time:.2f}s)")
            logging.info(f"   梯度提升: {self.model_performance['gb_accuracy']:.3f} (训练时间: {gb_train_time:.2f}s)")
            logging.info(f"   集成模型: {self.model_performance['ensemble_accuracy']:.3f}")

            return True

        except Exception as e:
            logging.error(f"❌ 模型训练失败: {e}")
            return False

    def get_current_features(self):
        """获取当前市场特征"""
        try:
            # 获取最新的50个数据点用于计算特征
            df = self.fetch_market_data(limit=50)
            if df is None:
                return None

            # 计算技术特征
            df_features = self.calculate_technical_features(df)

            # 获取最新的特征
            latest_features = df_features.iloc[-1]

            feature_columns = [
                'returns', 'log_returns',
                'price_vs_ma5', 'price_vs_ma10', 'price_vs_ma20',
                'ma5_vs_ma10', 'ma10_vs_ma20',
                'rsi', 'bb_position', 'volatility', 'price_range',
                'volume_ratio', 'price_volume',
                'momentum_5', 'momentum_10', 'trend_strength', 'price_position'
            ]

            # 检查是否有NaN值
            if latest_features[feature_columns].isna().any():
                logging.warning("⚠️ 特征包含NaN值，跳过此次预测")
                return None

            return latest_features[feature_columns].values

        except Exception as e:
            logging.error(f"❌ 获取当前特征失败: {e}")
            return None

    def predict(self, features):
        """进行预测"""
        try:
            if not self.is_trained:
                logging.warning("⚠️ 模型未训练，无法预测")
                return None, 0.0

            if features is None:
                return None, 0.0

            # 标准化特征
            features_scaled = self.scaler.transform([features])

            # 随机森林预测
            rf_proba = self.rf_model.predict_proba(features_scaled)[0]
            rf_pred = np.argmax(rf_proba)
            rf_confidence = max(rf_proba)

            # 梯度提升预测
            gb_proba = self.gb_model.predict_proba(features_scaled)[0]
            gb_pred = np.argmax(gb_proba)
            gb_confidence = max(gb_proba)

            # 集成决策
            if rf_pred == gb_pred:
                final_pred = rf_pred
                final_confidence = (rf_confidence + gb_confidence) / 2
            else:
                # 选择置信度更高的模型
                if rf_confidence > gb_confidence:
                    final_pred = rf_pred
                    final_confidence = rf_confidence
                else:
                    final_pred = gb_pred
                    final_confidence = gb_confidence

            # 只在高置信度时返回预测
            if final_confidence >= self.config['confidence_threshold']:
                return final_pred, final_confidence
            else:
                logging.info(f"🔍 置信度不足: {final_confidence:.3f} < {self.config['confidence_threshold']}")
                return None, final_confidence

        except Exception as e:
            logging.error(f"❌ 预测失败: {e}")
            return None, 0.0

    def evaluate_prediction(self, predicted_direction, actual_direction):
        """评估预测准确性"""
        if predicted_direction is not None and actual_direction is not None:
            self.total_predictions += 1
            if predicted_direction == actual_direction:
                self.correct_predictions += 1

            if self.total_predictions > 0 and self.total_predictions % 20 == 0:
                accuracy = self.correct_predictions / self.total_predictions
                logging.info(f"📊 当前预测准确率: {accuracy:.3f} ({self.correct_predictions}/{self.total_predictions})")

    def run_training_session(self):
        """运行训练会话"""
        logging.info("🚀 开始机器学习训练会话")

        # 获取历史数据
        df = self.fetch_market_data(limit=1000)
        if df is None:
            logging.error("❌ 无法获取训练数据")
            return False

        # 准备训练数据
        X, y = self.prepare_training_data(df)
        if X is None:
            logging.error("❌ 训练数据准备失败")
            return False

        # 训练模型
        success = self.train_models(X, y)

        if success:
            logging.info("🎉 训练会话成功完成!")
            return True
        else:
            logging.error("❌ 训练会话失败!")
            return False

    def run_prediction_session(self, duration_minutes=10):
        """运行预测会话"""
        if not self.is_trained:
            logging.error("❌ 模型未训练，请先运行训练会话")
            return

        logging.info(f"🔮 开始 {duration_minutes} 分钟预测会话")
        start_time = datetime.now()
        session_end = start_time + timedelta(minutes=duration_minutes)

        predictions_made = 0

        while datetime.now() < session_end:
            try:
                # 获取当前特征
                current_features = self.get_current_features()

                if current_features is not None:
                    # 进行预测
                    prediction, confidence = self.predict(current_features)

                    if prediction is not None:
                        predictions_made += 1
                        direction_text = "上涨" if prediction == 1 else "下跌/横盘"
                        logging.info(f"🎯 预测 #{predictions_made}: {direction_text} (置信度: {confidence:.3f})")

                # 等待一段时间
                time.sleep(30)  # 30秒间隔

            except KeyboardInterrupt:
                logging.info("🛑 用户手动停止预测会话")
                break
            except Exception as e:
                logging.error(f"❌ 预测循环错误: {e}")
                time.sleep(10)

        logging.info(f"🏁 预测会话完成! 总共进行了 {predictions_made} 次预测")

    def save_model(self, filename='ml_model_data.json'):
        """保存模型数据"""
        try:
            model_data = {
                'timestamp': datetime.now().isoformat(),
                'is_trained': self.is_trained,
                'model_performance': self.model_performance,
                'config': self.config,
                'prediction_stats': {
                    'total_predictions': self.total_predictions,
                    'correct_predictions': self.correct_predictions,
                    'accuracy': self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0
                }
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, indent=2, ensure_ascii=False)

            logging.info(f"💾 模型数据已保存到 {filename}")
            return True

        except Exception as e:
            logging.error(f"❌ 保存模型数据失败: {e}")
            return False

def main():
    """主函数 - 演示完整的训练和预测流程"""
    predictor = WorkingMLPredictor()

    try:
        print("="*60)
        print("🤖 真实机器学习价格预测引擎演示")
        print("="*60)

        # 1. 训练阶段
        print("\n📚 阶段1: 模型训练")
        training_success = predictor.run_training_session()

        if not training_success:
            print("❌ 训练失败，程序退出")
            return

        # 2. 预测阶段
        print("\n🔮 阶段2: 实时预测")
        predictor.run_prediction_session(duration_minutes=5)

        # 3. 保存结果
        print("\n💾 阶段3: 保存模型数据")
        predictor.save_model()

        print("\n" + "="*60)
        print("🎉 演示完成!")
        print("="*60)

    except Exception as e:
        logging.error(f"❌ 主程序运行失败: {e}")

if __name__ == "__main__":
    main()