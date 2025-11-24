#!/usr/bin/env python3
"""
模型评估和验证系统
专业的机器学习模型性能评估工具
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import logging
import json
from datetime import datetime

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型评估器"""

    def __init__(self):
        self.evaluation_results = {}
        self.config = {
            'cv_splits': 5,
            'test_size': 0.2,
            'random_state': 42
        }

    def basic_metrics(self, y_true, y_pred, y_prob=None):
        """基础分类指标"""
        metrics = {}

        # 基础指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')

        # 如果有概率预测，计算AUC
        if y_prob is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
            except:
                metrics['auc_roc'] = None

        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

        return metrics

    def time_series_split_validation(self, model, X, y):
        """时间序列分割验证"""
        logger.info("⏰ 开始时间序列分割验证")

        tscv = TimeSeriesSplit(n_splits=self.config['cv_splits'])
        cv_scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            cv_scores.append(score)

            logger.info(f"   Fold {fold + 1}: {score:.4f}")

        cv_results = {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'cv_folds': self.config['cv_splits']
        }

        logger.info(f"   平均分数: {cv_results['mean_score']:.4f} (±{cv_results['std_score']:.4f})")

        return cv_results

    def financial_metrics(self, y_true, y_pred, returns):
        """金融相关指标"""
        financial_metrics = {}

        # 只在预测正确的样本上计算收益
        correct_predictions = y_true == y_pred
        correct_returns = returns[correct_predictions]

        if len(correct_returns) > 0:
            financial_metrics['avg_return_correct'] = np.mean(correct_returns)
            financial_metrics['total_return_correct'] = np.prod(1 + correct_returns) - 1

        # 整体收益
        if len(returns) > 0:
            financial_metrics['avg_return_all'] = np.mean(returns)
            financial_metrics['total_return_all'] = np.prod(1 + returns) - 1

        # 夏普比率（年化）
        if len(returns) > 1:
            financial_metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 12)  # 假设5分钟数据

        # 最大回撤
        if len(returns) > 0:
            cumulative = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - peak) / peak
            financial_metrics['max_drawdown'] = np.min(drawdown)

        # 胜率
        financial_metrics['win_rate'] = np.mean(y_true == y_pred)

        return financial_metrics

    def detailed_classification_report(self, y_true, y_pred, target_names=['下跌/横盘', '上涨']):
        """详细的分类报告"""
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        return report

    def feature_importance_analysis(self, model, feature_names):
        """特征重要性分析"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = dict(zip(feature_names, importance))

            # 按重要性排序
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

            return sorted_importance
        else:
            return None

    def plot_confusion_matrix(self, y_true, y_pred, labels=['下跌/横盘', '上涨']):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, feature_importance, top_n=15):
        """绘制特征重要性"""
        if feature_importance:
            plt.figure(figsize=(10, 8))

            # 取前N个重要特征
            top_features = feature_importance[:top_n]
            features, importances = zip(*top_features)

            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('特征重要性')
            plt.title(f'Top {top_n} 特征重要性')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

    def comprehensive_evaluation(self, model, X, y, feature_names=None, returns=None):
        """综合评估"""
        logger.info("🔍 开始综合模型评估")

        # 分割数据
        split_idx = int(len(X) * (1 - self.config['test_size']))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
        else:
            y_prob = None

        # 基础指标
        basic_metrics = self.basic_metrics(y_test, y_pred, y_prob)

        # 时间序列交叉验证
        cv_results = self.time_series_split_validation(model, X, y)

        # 金融指标（如果有收益数据）
        if returns is not None and len(returns) == len(y):
            test_returns = returns[split_idx:]
            financial_metrics = self.financial_metrics(y_test, y_pred, test_returns)
        else:
            financial_metrics = {}

        # 详细分类报告
        detailed_report = self.detailed_classification_report(y_test, y_pred)

        # 特征重要性
        if feature_names is not None:
            feature_importance = self.feature_importance_analysis(model, feature_names)
        else:
            feature_importance = None

        # 汇总结果
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'test_size': len(y_test),
            'basic_metrics': basic_metrics,
            'cross_validation': cv_results,
            'financial_metrics': financial_metrics,
            'detailed_report': detailed_report,
            'feature_importance': feature_importance
        }

        # 打印结果
        self.print_evaluation_results(evaluation_results)

        return evaluation_results

    def print_evaluation_results(self, results):
        """打印评估结果"""
        print("\n" + "="*60)
        print("📊 模型评估结果")
        print("="*60)

        # 基础指标
        basic = results['basic_metrics']
        print(f"\n📈 基础指标:")
        print(f"   准确率 (Accuracy): {basic['accuracy']:.4f}")
        print(f"   精确率 (Precision): {basic['precision']:.4f}")
        print(f"   召回率 (Recall): {basic['recall']:.4f}")
        print(f"   F1分数: {basic['f1_score']:.4f}")
        if basic.get('auc_roc'):
            print(f"   AUC-ROC: {basic['auc_roc']:.4f}")

        # 交叉验证
        cv = results['cross_validation']
        print(f"\n⏰ 时间序列交叉验证:")
        print(f"   平均分数: {cv['mean_score']:.4f} ± {cv['std_score']:.4f}")
        print(f"   CV分数: {[f'{score:.4f}' for score in cv['cv_scores']]}")

        # 金融指标
        if results['financial_metrics']:
            fin = results['financial_metrics']
            print(f"\n💰 金融指标:")
            print(f"   胜率: {fin['win_rate']:.4f}")
            if fin.get('avg_return_correct'):
                print(f"   正确预测平均收益: {fin['avg_return_correct']:.4f}")
            if fin.get('sharpe_ratio'):
                print(f"   夏普比率: {fin['sharpe_ratio']:.4f}")
            if fin.get('max_drawdown'):
                print(f"   最大回撤: {fin['max_drawdown']:.4f}")

        # 特征重要性
        if results['feature_importance']:
            print(f"\n🔍 Top 10 重要特征:")
            for i, (feature, importance) in enumerate(results['feature_importance'][:10], 1):
                print(f"   {i:2d}. {feature:<20} {importance:.4f}")

        print("\n" + "="*60)

    def save_evaluation_results(self, results, filename='model_evaluation.json'):
        """保存评估结果"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 评估结果已保存到 {filename}")
            return True
        except Exception as e:
            logger.error(f"❌ 保存评估结果失败: {e}")
            return False

def demo_evaluation():
    """演示模型评估"""
    print("🔍 模型评估系统演示")
    print("="*50)

    # 创建模拟数据
    print("\n📊 创建模拟数据...")
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # 模拟特征
    X = np.random.randn(n_samples, n_features)

    # 模拟标签（有一定逻辑性）
    y = (X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)

    # 模拟收益率
    returns = np.random.randn(n_samples) * 0.001

    # 特征名称
    feature_names = [f'feature_{i+1}' for i in range(n_features)]

    print(f"✅ 数据创建完成: {n_samples} 样本, {n_features} 特征")
    print(f"   标签分布: {np.bincount(y)}")
    print(f"   收益率范围: [{returns.min():.4f}, {returns.max():.4f}]")

    # 创建评估器
    from sklearn.ensemble import RandomForestClassifier
    evaluator = ModelEvaluator()

    # 创建模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 运行评估
    print("\n🔍 开始模型评估...")
    results = evaluator.comprehensive_evaluation(model, X, y, feature_names, returns)

    # 绘制混淆矩阵
    print("\n📊 绘制混淆矩阵...")
    evaluator.plot_confusion_matrix(
        results['basic_metrics']['confusion_matrix'],
        labels=['下跌/横盘', '上涨']
    )

    # 绘制特征重要性
    print("\n📊 绘制特征重要性...")
    evaluator.plot_feature_importance(results['feature_importance'])

    # 保存结果
    print("\n💾 保存评估结果...")
    evaluator.save_evaluation_results(results)

    print("\n🎉 模型评估演示完成!")

if __name__ == "__main__":
    demo_evaluation()