# graphical_data
feed graphical data as input data

## dqn_graph_relative_price.py
1. 2011~2015 train, 2016 test
2. feed relative price line chart as input
3. train with 2 actions, 0 represents "do nothing", and 1 represents "buy and then sell"
4. rewards of day t = (close price of day t) - (close price of day (t-1))

## dqn_graph_ma.py
1. 2011~2015 train, 2016 test
2. feed relative MA5, MA20 line chart as input
   ( relative MA of day t = ((MA of day t) - (MA of day (t-1))) / (MA of day (t-1)) )
3. train with 2 actions, 0 represents "do nothing", and "1" represents "buy and then sell"
4. rewards of day t = (close price of day t) - (close price of day (t-1))
