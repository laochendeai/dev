# Gate.io机器学习交易系统

基于机器学习的Gate.io期货自动化交易系统，支持实盘测试网环境。

## 快速启动

### 安装依赖
```bash
pip install -r requirements.txt
```

### 配置API密钥
```bash
export GATEIO_API_KEY="your_api_key"
export GATEIO_API_SECRET="your_secret"
export GATEIO_TESTNET=true
```

### 启动交易系统
```bash
python deploy_real_gateio_trading.py
```

## 系统特性

- 🧠 **机器学习预测**: 基于随机森林的价格预测模型
- ⚡ **实时交易**: 每30秒分析市场并自动下单
- 🛡️ **风险控制**: 内置止盈止损机制
- 📊 **实时监控**: 完整的交易日志和性能统计
- 🌐 **测试网支持**: 在Gate.io测试网环境安全交易

## 查看委托

所有真实委托可在Gate.io测试网页面查看：
- 现货委托: https://testnet.gate.com/orders
- 期货委托: https://testnet.gate.com/futures/USDT/ETH_USDT

## 获取测试币

测试网USDT和ETH: https://testnet.gate.com/faucet

## 重要提示

- 本系统运行在测试网环境，不会产生真实资金损失
- 请确保账户有足够的测试币进行交易
- 建议先用小额资金测试系统稳定性