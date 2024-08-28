# Trading by Indicators

## STRATEGY
- [ ] Signal processing of RSI thresholds -> 04-signal.py
- [ ] Use of dynamic position size: Chaikin volatility index, high volatility environments should have higher target percent values.
- [ ] Preexisting position is closed if target percent is met or liquidation threshold or opposite signal is triggered.
- [ ] Multiple positions of same directionality or opposing directionality not decided yet 
- [ ] Parameters of search space: target_percent, risk_per_trade, leverage, rsi_oversold, rsi_overbought, intervall_step

## TODO
- [ ] dynamic target percent, leverage, risk per trade as a function of Chaikin Volatility Index
- [ ] implement API
- [ ] more robust statistical testing in 01-bruteforce.py
