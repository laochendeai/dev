#!/usr/bin/env python3
"""
机器学习价格预测模块
基于深度学习的超短期价格预测（目标：85%+准确率）
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import ccxt
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
        logging.FileHandler('ml_price_predictor.log'),
        logging.StreamHandler()
    ]
)

class MLPricePredictor:
    """机器学习价格预测器"""

    def __init__(self):
        # 模型配置
        self.config = {
            'sequence_length': 50,        # 序列长度
            'prediction_horizon': 3,      # 预测步长 (3个5分钟K线 = 15分钟)
            'feature_count': 15,          # 特征数量
            'model_update_freq': 1000,    # 模型更新频率
            'confidence_threshold': 0.7,  # 置信度阈值
            'retrain_threshold': 0.65     # 重训练阈值
        }

        # 数据存储
        self.price_data = deque(maxlen=10000)
        self.order_book_data = deque(maxlen=5000)
        self.feature_data = deque(maxlen=5000)
        self.labels = deque(maxlen=5000)

        # 模型初始化
        self.scaler = MinMaxScaler()
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.lstm_model = None
        self.model_ensemble = {}

        # 性能跟踪
        self.predictions = []
        self.correct_predictions = 0
        self.total_predictions = 0
        self.model_performance = {
            'rf_accuracy': 0,
            'gb_accuracy': 0,
            'lstm_accuracy': 0,
            'ensemble_accuracy': 0
        }

        # 交易所连接
        self.exchange = ccxt.binance()

        logging.info("🤖 机器学习价格预测器已初始化")
        logging.info(f"📊 配置: 序列长度={self.config['sequence_length']}, 预测步长={self.config['prediction_horizon']}")

    def fetch_market_data(self):
        """获取市场数据"""
        try:
            # 获取K线数据
            ohlcv = self.exchange.fetch_ohlcv('ETH/USDT', '1m', limit=200)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # 获取订单簿数据
            orderbook = self.exchange.fetch_order_book('ETH/USDT', limit=20)
            lob_data = {
                'timestamp': datetime.now(),
                'bid_prices': [float(b[0]) for b in orderbook['bids'][:10]],
                'bid_volumes': [float(b[1]) for b in orderbook['bids'][:10]],
                'ask_prices': [float(a[0]) for a in orderbook['asks'][:10]],
                'ask_volumes': [float(a[1]) for a in orderbook['asks'][:10]]
            }

            return df, lob_data

        except Exception as e:
            logging.error(f"获取市场数据失败: {e}")
            return None, None

    def calculate_technical_indicators(self, df):
        """计算技术指标"""
        if len(df) < 20:
            return df

        df = df.copy()

        # 价格相关指标
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=10).std()
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # 布林带
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # 成交量指标
        df['volume_ma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        return df

    def calculate_order_book_features(self, lob_data):
        """计算订单簿特征"""
        features = []

        try:
            # 1. 订单簿不平衡
            bid_volume = sum(lob_data['bid_volumes'][:5])
            ask_volume = sum(lob_data['ask_volumes'][:5])
            if (bid_volume + ask_volume) > 0:
                imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            else:
                imbalance = 0
            features.append(imbalance)

            # 2. 价差特征
            if lob_data['bid_prices'] and lob_data['ask_prices']:
                mid_price = (lob_data['bid_prices'][0] + lob_data['ask_prices'][0]) / 2
                spread = lob_data['ask_prices'][0] - lob_data['bid_prices'][0]
                spread_ratio = spread / mid_price
            else:
                spread_ratio = 0
            features.append(spread_ratio)

            # 3. 流动性深度比率
            total_bid = sum(lob_data['bid_volumes'])
            total_ask = sum(lob_data['ask_volumes'])
            if total_ask > 0:
                liquidity_ratio = total_bid / total_ask
            else:
                liquidity_ratio = 1
            features.append(min(liquidity_ratio, 5))  # 限制最大值

            # 4. 订单簿斜率
            if len(lob_data['bid_prices']) > 1:
                bid_slope = (lob_data['bid_prices'][-1] - lob_data['bid_prices'][0]) / len(lob_data['bid_prices'])
                ask_slope = (lob_data['ask_prices'][-1] - lob_data['ask_prices'][0]) / len(lob_data['ask_prices'])
                slope_diff = bid_slope - ask_slope
            else:
                slope_diff = 0
            features.append(slope_diff)

            # 5. VWAP偏离
            if lob_data['bid_prices'] and lob_data['ask_prices']:
                total_volume = 0
                weighted_sum = 0

                for price, volume in zip(lob_data['bid_prices'], lob_data['bid_volumes']):
                    weighted_sum += price * volume
                    total_volume += volume

                for price, volume in zip(lob_data['ask_prices'], lob_data['ask_volumes']):
                    weighted_sum += price * volume
                    total_volume += volume

                if total_volume > 0:
                    vwap = weighted_sum / total_volume
                    mid_price = (lob_data['bid_prices'][0] + lob_data['ask_prices'][0]) / 2
                    vwap_deviation = (vwap - mid_price) / mid_price
                else:
                    vwap_deviation = 0
            else:
                vwap_deviation = 0
            features.append(vwap_deviation)

        except Exception as e:
            logging.error(f"订单簿特征计算失败: {e}")
            features.extend([0] * 5)  # 返回默认值

        return features

    def create_features_and_labels(self, df, lob_data):
        """创建特征和标签"""
        try:
            if len(df) < self.config['sequence_length'] + self.config['prediction_horizon']:
                return None, None

            # 计算技术指标
            df = self.calculate_technical_indicators(df)

            # 选择特征列
            feature_columns = [
                'returns', 'volatility', 'ma_5', 'ma_10', 'ma_20',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_width', 'volume_ratio'
            ]

            # 计算最新价格特征
            latest_features = []
            for col in feature_columns:
                if col in df.columns and not pd.isna(df[col].iloc[-1]):
                    latest_features.append(df[col].iloc[-1])
                else:
                    latest_features.append(0)

            # 添加订单簿特征
            lob_features = self.calculate_order_book_features(lob_data)
            latest_features.extend(lob_features)

            # 创建标签（未来价格方向）
            current_price = df['close'].iloc[-1]
            future_prices = df['close'].iloc[-self.config['prediction_horizon']:]

            if len(future_prices) < self.config['prediction_horizon']:
                return None, None

            # 计算未来价格变化
            max_future_price = max(future_prices)
            min_future_price = min(future_prices)
            avg_future_price = np.mean(future_prices)

            # 标签定义：1=上涨，0=下跌，-1=横盘
            price_change_threshold = current_price * 0.001  # 0.1%阈值

            if max_future_price > current_price + price_change_threshold:
                if avg_future_price > current_price + price_change_threshold/2:
                    label = 1  # 上涨
                elif avg_future_price < current_price - price_change_threshold/2:
                    label = 0  # 下跌
                else:
                    label = -1  # 横盘
            elif min_future_price < current_price - price_change_threshold:
                label = 0  # 下跌
            else:
                label = -1  # 横盘

            return latest_features, label

        except Exception as e:
            logging.error(f"特征标签创建失败: {e}")
            return None, None

    def train_random_forest(self, X, y):
        """训练随机森林模型"""
        try:
            # 过滤掉横盘标签
            mask = y != -1
            X_filtered = X[mask]
            y_filtered = y[mask]

            if len(X_filtered) < 50:  # 数据不足
                return False

            self.rf_model.fit(X_filtered, y_filtered)
            return True

        except Exception as e:
            logging.error(f"随机森林训练失败: {e}")
            return False

    def train_gradient_boosting(self, X, y):
        """训练梯度提升模型"""
        try:
            mask = y != -1
            X_filtered = X[mask]
            y_filtered = y[mask]

            if len(X_filtered) < 50:
                return False

            self.gb_model.fit(X_filtered, y_filtered)
            return True

        except Exception as e:
            logging.error(f"梯度提升训练失败: {e}")
            return False

    def create_lstm_model(self):
        """创建LSTM模型"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(self.config['sequence_length'], self.config['feature_count'])),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(32, return_sequences=False),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(3, activation='softmax')  # 3个类别：上涨、下跌、横盘
            ])

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return model

        except Exception as e:
            logging.error(f"LSTM模型创建失败: {e}")
            return None

    def train_lstm(self, X_sequences, y_sequences):
        """训练LSTM模型"""
        try:
            if self.lstm_model is None:
                self.lstm_model = self.create_lstm_model()

            if len(X_sequences) < 100:  # 数据不足
                return False

            # 数据预处理
            X_scaled = self.scaler.fit_transform(X_sequences.reshape(-1, X_sequences.shape[-1])).reshape(X_sequences.shape)

            # 训练
            self.lstm_model.fit(X_scaled, y_sequences, epochs=10, batch_size=32, verbose=0)
            return True

        except Exception as e:
            logging.error(f"LSTM训练失败: {e}")
            return False

    def ensemble_predict(self, features):
        """集成预测"""
        try:
            predictions = {}
            confidences = {}

            # 随机森林预测
            try:
                rf_pred = self.rf_model.predict_proba([features])[0]
                if len(rf_pred) == 2:
                    predictions['rf'] = np.argmax(rf_pred)
                    confidences['rf'] = max(rf_pred)
                else:
                    predictions['rf'] = -1
                    confidences['rf'] = 0
            except:
                predictions['rf'] = -1
                confidences['rf'] = 0

            # 梯度提升预测
            try:
                gb_pred = self.gb_model.predict_proba([features])[0]
                if len(gb_pred) == 2:
                    predictions['gb'] = np.argmax(gb_pred)
                    confidences['gb'] = max(gb_pred)
                else:
                    predictions['gb'] = -1
                    confidences['gb'] = 0
            except:
                predictions['gb'] = -1
                confidences['gb'] = 0

            # LSTM预测（如果模型存在）
            try:
                if self.lstm_model is not None and len(self.feature_data) >= self.config['sequence_length']:
                    # 创建序列
                    recent_features = list(self.feature_data)[-self.config['sequence_length']:]
                    features_array = np.array(recent_features)
                    features_scaled = self.scaler.transform(features_array.reshape(-1, features_array.shape[-1])).reshape(features_array.shape)

                    lstm_pred = self.lstm_model.predict(features_array.reshape(1, self.config['sequence_length'], -1), verbose=0)[0]
                    lstm_class = np.argmax(lstm_pred)
                    lstm_confidence = max(lstm_pred)

                    # 转换为二分类（忽略横盘）
                    if lstm_class == 2:  # 横盘
                        predictions['lstm'] = -1
                        confidences['lstm'] = lstm_confidence
                    else:
                        predictions['lstm'] = lstm_class
                        confidences['lstm'] = lstm_confidence
                else:
                    predictions['lstm'] = -1
                    confidences['lstm'] = 0
            except:
                predictions['lstm'] = -1
                confidences['lstm'] = 0

            # 加权集成
            valid_predictions = {k: v for k, v in predictions.items() if v != -1}
            valid_confidences = {k: confidences[k] for k in valid_predictions.keys()}

            if not valid_predictions:
                return -1, 0

            # 权重分配
            weights = {'rf': 0.3, 'gb': 0.3, 'lstm': 0.4}
            total_weight = 0
            weighted_sum = 0
            total_confidence = 0

            for model, pred in valid_predictions.items():
                weight = weights.get(model, 0.33)
                confidence = valid_confidences.get(model, 0.5)
                adjusted_weight = weight * confidence

                weighted_sum += pred * adjusted_weight
                total_weight += adjusted_weight
                total_confidence += confidence

            if total_weight > 0:
                ensemble_pred = int(round(weighted_sum / total_weight))
                ensemble_confidence = total_confidence / len(valid_predictions)
            else:
                ensemble_pred = -1
                ensemble_confidence = 0

            return ensemble_pred, ensemble_confidence

        except Exception as e:
            logging.error(f"集成预测失败: {e}")
            return -1, 0

    def update_models(self):
        """更新模型"""
        try:
            if len(self.feature_data) < 100:  # 数据不足
                return False

            # 准备数据
            X = np.array(list(self.feature_data))
            y = np.array(list(self.labels))

            # 过滤掉横盘数据
            mask = y != -1
            X_filtered = X[mask]
            y_filtered = y[mask]

            if len(X_filtered) < 50:
                return False

            # 训练模型
            rf_success = self.train_random_forest(X_filtered, y_filtered)
            gb_success = self.train_gradient_boosting(X_filtered, y_filtered)

            logging.info(f"🤖 模型更新完成: RF={rf_success}, GB={gb_success}, 数据量={len(X_filtered)}")
            return True

        except Exception as e:
            logging.error(f"模型更新失败: {e}")
            return False

    def predict_price_direction(self, current_features):
        """预测价格方向"""
        try:
            # 集成预测
            prediction, confidence = self.ensemble_predict(current_features)

            # 记录预测
            self.predictions.append({
                'timestamp': datetime.now(),
                'prediction': prediction,
                'confidence': confidence,
                'features': current_features
            })

            # 更新统计
            if prediction != -1:  # 排除横盘
                self.total_predictions += 1

            return prediction, confidence

        except Exception as e:
            logging.error(f"价格预测失败: {e}")
            return -1, 0

    def evaluate_prediction(self, predicted_direction, actual_direction):
        """评估预测准确性"""
        try:
            if predicted_direction == -1 or actual_direction == -1:
                return  # 忽略横盘

            if predicted_direction == actual_direction:
                self.correct_predictions += 1

            # 计算当前准确率
            if self.total_predictions > 0:
                current_accuracy = self.correct_predictions / self.total_predictions
                logging.info(f"📊 当前预测准确率: {current_accuracy:.1%} ({self.correct_predictions}/{self.total_predictions})")

                # 如果准确率过低，触发模型重训练
                if current_accuracy < self.config['retrain_threshold'] and self.total_predictions % 50 == 0:
                    logging.warning("🔄 预测准确率过低，触发模型重训练")
                    self.update_models()

        except Exception as e:
            logging.error(f"预测评估失败: {e}")

    def run_prediction_session(self, duration_minutes=20):
        """运行预测会话"""
        logging.info(f"🔮 开始 {duration_minutes} 分钟机器学习预测会话")
        logging.info("="*60)

        start_time = datetime.now()
        session_end = start_time + timedelta(minutes=duration_minutes)
        data_count = 0

        while datetime.now() < session_end:
            try:
                # 获取市场数据
                df, lob_data = self.fetch_market_data()
                if df is None or lob_data is None:
                    time.sleep(5)
                    continue

                # 创建特征和标签
                features, label = self.create_features_and_labels(df, lob_data)
                if features is None or label is None:
                    time.sleep(5)
                    continue

                # 存储数据
                self.feature_data.append(features)
                self.labels.append(label)

                data_count += 1

                # 定期更新模型
                if data_count % 100 == 0 and len(self.feature_data) >= 100:
                    self.update_models()

                # 生成预测
                if len(self.feature_data) >= 50:  # 有足够的历史数据
                    prediction, confidence = self.predict_price_direction(features)

                    if prediction != -1 and confidence >= self.config['confidence_threshold']:
                        direction_text = "上涨" if prediction == 1 else "下跌"
                        logging.info(f"🎯 强信号预测: {direction_text} (置信度: {confidence:.1%})")

                    # 模拟验证（在实际应用中，需要等待真实结果）
                    if len(self.predictions) > 10 and data_count % 20 == 0:
                        # 简单的模拟验证（实际应用中需要真实价格验证）
                        recent_predictions = self.predictions[-5:]
                        correct = sum(1 for p in recent_predictions if p.get('verified', False))
                        logging.info(f"📈 预测状态: {len(self.predictions)} 个预测，最新验证: {correct}/5")

                time.sleep(60)  # 每分钟更新一次

            except KeyboardInterrupt:
                logging.info("🛑 用户手动停止预测会话")
                break
            except Exception as e:
                logging.error(f"❌ 预测循环错误: {e}")
                time.sleep(10)

        # 最终统计
        logging.info("="*60)
        logging.info("🏁 预测会话结束！")
        logging.info(f"📊 会话统计:")
        logging.info(f"   数据点数: {data_count}")
        logging.info(f"   特征数据量: {len(self.feature_data)}")
        logging.info(f"   总预测数: {len(self.predictions)}")
        logging.info(f"   有效预测数: {self.total_predictions}")

        if self.total_predictions > 0:
            accuracy = self.correct_predictions / self.total_predictions
            logging.info(f"   预测准确率: {accuracy:.1%}")
            logging.info(f"   正确预测: {self.correct_predictions}")

            if accuracy >= 0.8:
                logging.info("🎉 达到80%+预测准确率目标！")
            elif accuracy >= 0.7:
                logging.info("🟡 接近目标，预测准确率70%+")
            else:
                logging.info("🔴 需要进一步优化模型")

        return {
            'data_points': data_count,
            'predictions': len(self.predictions),
            'accuracy': self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0
        }

def main():
    """主函数"""
    try:
        predictor = MLPricePredictor()

        # 运行20分钟预测会话
        results = predictor.run_prediction_session(duration_minutes=20)

        # 保存结果
        results_file = 'ml_prediction_results.json'
        save_data = {
            'session_time': datetime.now().isoformat(),
            'config': predictor.config,
            'performance': results,
            'model_performance': predictor.model_performance
        }

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        logging.info(f"📁 预测结果已保存到 {results_file}")

    except Exception as e:
        logging.error(f"主程序运行失败: {e}")

if __name__ == "__main__":
    main()