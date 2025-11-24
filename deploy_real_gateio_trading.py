#!/usr/bin/env python3
"""
启动支持真实Gate.io测试网委托的ML交易系统
现在会在Gate.io测试网页上显示真实的委托订单
"""

import time
import logging
from datetime import datetime
from gateio_ml_predictor import WorkingMLPredictor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gateio_real_trading.log'),
        logging.StreamHandler()
    ]
)

def main():
    """启动Gate.io真实测试网交易会话"""
    print("🚀 Gate.io真实测试网ML交易系统启动")
    print("="*60)
    print("📋 重要提示：")
    print("   • 本系统现在会在Gate.io测试网提交真实订单")
    print("   • 您可以在 https://testnet.gate.com 查看委托记录")
    print("   • 这是测试网环境，不会产生真实资金损失")
    print("   • 如果账户没有测试币，请先获取测试币")
    print("="*60)

    # API配置
    API_KEY = "edc886fb2bc311593abc07803d5123a7"
    SECRET = "c4f20bacd9e9a73e3fd4a580052982f7a4f0cd86f6d18bc890a60f01d3ac4d68"
    SYMBOL = 'ETH/USDT'

    try:
        # 初始化交易系统
        print("🎯 初始化Gate.io真实测试网交易系统...")
        predictor = WorkingMLPredictor(API_KEY, SECRET, testnet=True)

        print(f"✅ 系统初始化成功")
        print(f"🌐 交易所: {predictor.exchange_name}")
        print(f"🧪 环境: 真实测试网")
        print(f"💎 交易对: {SYMBOL}")
        print(f"🔗 查看委托: https://testnet.gate.com")

        # 检查API连接
        if not predictor.adapter.test_connection():
            print("❌ API连接失败，请检查网络和API配置")
            return

        print("✅ API连接正常")

        # 检查余额
        print(f"\n💰 检查测试网余额...")
        balance = predictor.adapter.get_balance()

        if balance and 'total' in balance:
            usdt_balance = float(balance['total'].get('USDT', 0))
            eth_balance = float(balance['total'].get('ETH', 0))

            print(f"💵 USDT余额: {usdt_balance}")
            print(f"💎 ETH余额: {eth_balance}")

            if usdt_balance < 30 and eth_balance < 0.01:
                print(f"\n⚠️ 测试网余额不足！")
                print(f"💡 10倍合约要求: 最小27.99 USDT ≈ 0.01 ETH")
                print(f"📖 获取测试币步骤：")
                print(f"   1. 访问 https://testnet.gate.com/")
                print(f"   2. 注册或登录测试账户")
                print(f"   3. 点击 '获取测试币' 按钮")
                print(f"   4. 获取USDT和ETH测试币 (建议至少100 USDT)")
                print(f"   5. 重新运行本程序")
                return
            else:
                print(f"✅ 余额充足，可以开始交易")
        else:
            print(f"⚠️ 无法获取余额信息，但会继续尝试交易")

        # 训练模型（如果尚未训练）
        if not predictor.is_trained:
            print(f"\n🧠 开始训练ML模型...")
            training_success = predictor.run_training_session(SYMBOL)

            if not training_success:
                print("❌ 模型训练失败，无法继续")
                return

            print("✅ 模型训练完成!")
        else:
            print("✅ 模型已训练，直接开始预测")

        # 获取当前市场信息
        print(f"\n📊 当前市场信息 ({SYMBOL}):")
        ticker = predictor.adapter.get_ticker(SYMBOL)
        if ticker:
            print(f"💰 当前价格: ${ticker['last']:.2f}")
            print(f"📈 24h变化: {ticker.get('percentage', 0):+.2f}%")
            print(f"💵 买一价: ${ticker['bid']:.2f}")
            print(f"💵 卖一价: ${ticker['ask']:.2f}")

        # 显示系统状态
        print(f"\n📈 系统状态:")
        print(f"🤖 ML模型: {'已训练' if predictor.is_trained else '未训练'}")
        print(f"🎯 集成准确率: {predictor.model_performance['ensemble_accuracy']:.1%}")
        print(f"⚡ 置信度阈值: {predictor.config['confidence_threshold']:.1%}")

        print(f"\n🚀 开始实时预测和真实测试网交易...")
        print(f"⏰ 预测间隔: 30秒")
        print(f"🔄 按 Ctrl+C 停止交易会话")
        print(f"🌐 查看委托: https://testnet.gate.com/trade/{SYMBOL}")
        print("-" * 60)

        # 实时交易循环
        session_start = datetime.now()
        predictions_made = 0
        real_trades_executed = 0

        try:
            while True:
                current_time = datetime.now()
                elapsed = (current_time - session_start).total_seconds() / 60

                # 获取当前特征
                current_features = predictor.get_current_features(SYMBOL)

                if current_features is not None:
                    # 进行ML预测
                    prediction, confidence = predictor.predict(current_features, SYMBOL)
                    predictions_made += 1

                    # 显示时间戳
                    print(f"\n⏰ {current_time.strftime('%H:%M:%S')} | 运行 {elapsed:.1f} 分钟")

                    if prediction is not None:
                        direction = "📈 上涨" if prediction == 1 else "📉 下跌/横盘"
                        print(f"🎯 ML预测: {direction} (置信度: {confidence:.1%})")

                        # 获取实时价格
                        ticker = predictor.adapter.get_ticker(SYMBOL)
                        if ticker:
                            current_price = ticker['last']

                            # 执行真实测试网交易策略
                            if prediction == 1:  # 预测上涨 - 买入
                                order_price = ticker['bid'] * 0.995  # 略低于买一价
                                order = predictor.adapter.place_order(
                                    symbol=SYMBOL,
                                    order_type='limit',
                                    side='buy',
                                    amount=0.01,  # 满足10倍合约最小要求 (≈27.99 USDT)
                                    price=order_price
                                )

                                if order and not order.get('mock', False):
                                    real_trades_executed += 1
                                    print(f"🛒 ✅ 真实买单已提交!")
                                    print(f"   订单ID: {order.get('id', 'N/A')}")
                                    print(f"   数量: 0.01 ETH (满足10倍合约最小要求)")
                                    print(f"   价格: ${order_price:.2f}")
                                    print(f"   价值: ≈${0.01 * order_price:.2f}")
                                    print(f"   🔗 在线查看: https://testnet.gate.com")
                                elif order and order.get('mock'):
                                    print(f"🧪 模拟买单: 真实下单失败，使用模拟")
                                else:
                                    print(f"❌ 买单失败")

                            else:  # 预测下跌/横盘 - 卖出
                                order_price = ticker['ask'] * 1.005  # 略高于卖一价
                                order = predictor.adapter.place_order(
                                    symbol=SYMBOL,
                                    order_type='limit',
                                    side='sell',
                                    amount=0.01,  # 满足10倍合约最小要求 (≈27.99 USDT)
                                    price=order_price
                                )

                                if order and not order.get('mock', False):
                                    real_trades_executed += 1
                                    print(f"💰 ✅ 真实卖单已提交!")
                                    print(f"   订单ID: {order.get('id', 'N/A')}")
                                    print(f"   数量: 0.01 ETH (满足10倍合约最小要求)")
                                    print(f"   价格: ${order_price:.2f}")
                                    print(f"   价值: ≈${0.01 * order_price:.2f}")
                                    print(f"   🔗 在线查看: https://testnet.gate.com")
                                elif order and order.get('mock'):
                                    print(f"🧪 模拟卖单: 真实下单失败，使用模拟")
                                else:
                                    print(f"❌ 卖单失败")

                    else:
                        print(f"🔍 置信度不足 ({confidence:.1%} < {predictor.config['confidence_threshold']:.1%})，跳过此次交易")

                    # 更新价格信息
                    ticker = predictor.adapter.get_ticker(SYMBOL)
                    if ticker:
                        print(f"💰 当前价格: ${ticker['last']:.2f} | 24h: {ticker.get('percentage', 0):+.2f}%")

                    # 显示统计信息
                    print(f"📊 会话统计: 预测 {predictions_made} 次, 真实交易 {real_trades_executed} 笔")

                else:
                    print(f"⚠️ 无法获取市场特征，跳过此次预测")

                # 等待30秒进行下一次预测
                time.sleep(30)

        except KeyboardInterrupt:
            print(f"\n\n🛑 用户手动停止交易会话")

            session_end = datetime.now()
            total_time = (session_end - session_start).total_seconds() / 60

            print(f"\n📋 交易会话总结:")
            print(f"⏰ 运行时间: {total_time:.1f} 分钟")
            print(f"🎯 总预测次数: {predictions_made}")
            print(f"💼 总真实交易次数: {real_trades_executed}")
            print(f"📈 预测频率: {predictions_made/total_time:.1f} 次/分钟" if total_time > 0 else "N/A")
            print(f"💼 交易频率: {real_trades_executed/total_time:.1f} 笔/分钟" if total_time > 0 else "N/A")

            print(f"\n🌐 查看所有委托: https://testnet.gate.com/orders")
            print(f"📊 查看交易历史: https://testnet.gate.com/history")

            # 获取最终价格
            final_ticker = predictor.adapter.get_ticker(SYMBOL)
            if final_ticker:
                print(f"💰 结束时价格: ${final_ticker['last']:.2f}")

            print(f"\n💾 保存会话数据...")
            predictor.save_model('gateio_real_trading_end.json')
            print(f"✅ 会话数据已保存")

            print(f"\n🎉 Gate.io真实测试网交易会话结束!")
            print(f"📄 详细日志保存在: gateio_real_trading.log")

    except Exception as e:
        logging.error(f"❌ 交易系统运行错误: {e}")
        print(f"❌ 系统错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()