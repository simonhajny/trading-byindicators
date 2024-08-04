# trading-byindicators
The trading logic consists of opening a position "long" or "short" if RSI thresholds reach defined thresholds. The position is sold after the price gain or fall is a predefined magnitude. 
Liquidations are taken into account and amount to a loss equal to the initial size of the position. Additional parameters include, leverage, risk per trade as the percentage of entire balance used for each trade and intervall length. 

