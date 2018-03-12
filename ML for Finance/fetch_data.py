#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 12:58:43 2018

@author: user
"""

import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt

stock=['AAPL']

start_date='01/01/2001'
end_date='01/01/2018'

data=web.DataReader(stock, data_source='yahoo', start=start_date, end=end_date)['Adj Close']
daily_returns=(data/data.shift(1))-1
daily_returns.hist(bins=100)
plt.show()