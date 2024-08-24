# Trading by Indicators

## STRATEGY
- [ ] Uses Metropolis for search space optimization.
- [ ] Core signal given by RSI thresholds
- [ ] Parameters of search space: target_percent, leverage, rsi_oversold, rsi_overboughty, risk_per_trade, intervall_step

## TODO
- [ ] add partial_take_profit parameter
- [ ] add additional core signal parameters: ATR (average true range, bollinger bands, MACD)
- [ ] implement API in 01-algorithmic.py
- [ ] more robust statistical testing in 01-backtest-bruteforce.py
