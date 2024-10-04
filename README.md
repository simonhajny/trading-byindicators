# Trading by Indicators

## 03-SIGNAL-PLOT.py
- [ ] contains 12 diffrent RSI types, 3 diffrent windows RSI, 3 composite RSIs (diffrent ways of adding the 3 windows) and for each of the previous 6 RSIs a composition with Chaikin Volatility index gives another 6 RSIs
- [ ] the parameters are: ALPHA, BETA, LONGTHRESHOLD, SHORTHRESHOLD; the first two determine the nature of the composite of CVI and RSIs, whereas the last two are criteria for trade signal logging
- [ ] contains a optimization class which optimizes accuracy for the previously mentioned parameters

## REALTIME.py
- [ ] accesses signals
- [ ] the parametersare: position_size, leverage, target_percent, stop_loss percent
