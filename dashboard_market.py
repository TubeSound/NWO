import time
import threading
import numpy as np
import pandas as pd
from dateutil import tz
from datetime import datetime, timedelta, timezone
from mt5_trade import Mt5Trade, Columns, PositionInfo

import streamlit as st

from time_utils import TimeUtils
from data_buffer import DataBuffer
from candle_chart import CandleChart


JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')  

SYMBOLS = ['', 'NIKKEI', 'DOW', 'NSDQ']
TIMEFRAMES = ['M5', 'M15', 'M30', 'H1', 'H4', 'D1']


def calc_indicators(timeframe, data, technical_params):
    pass
    

class Mt5Manager:
    def __init__(self, symbol, timeframe):
        self.set_sever_time()
        self.mt5 = Mt5Trade(symbol)
        self.mt5.connect()
        self.symbol = symbol
        self.timeframe = timeframe
    
    def utcnow(self):
        #utc1 = datetime.utcnow()
        #utc1 = utc1.replace(tzinfo=UTC)
        utc = datetime.now(UTC)
        return utc

    def utc2localize(self, aware_utc_time, timezone):
        t = aware_utc_time.astimezone(timezone)
        return t

    def is_market_open(self, mt5, timezone):
        now = self.utcnow()
        t = self.utc2localize(now, timezone)
        t -= timedelta(seconds=5)
        df = mt5.get_ticks_from(t, length=100)
        return (len(df) > 0)
        
    def wait_market_open(self, mt5, timezone):
        while self.is_market_open(mt5, timezone) == False:
            time.sleep(5)
    
    def set_sever_time(self):
        begin_month = 3
        begin_sunday = 2
        end_month = 11
        end_sunday = 1
        delta_hour_from_gmt_in_summer = 3
        now = datetime.now(JST)
        dt, tz = TimeUtils.delta_hour_from_gmt(now, begin_month, begin_sunday, end_month, end_sunday, delta_hour_from_gmt_in_summer)
        self.delta_hour_from_gmt  = dt
        self.server_timezone = tz
        print('SeverTime GMT+', dt, tz)
        
    def init(self, length, technical_function, technical_param, remove_last=False):
        df = self.mt5.get_rates(self.timeframe, length)
        if remove_last:
            df = df.iloc[:-1, :]
        buffer = DataBuffer(technical_function, self.symbol, self.timeframe, df, technical_param, self.delta_hour_from_gmt)
        self.buffer = buffer
        return buffer.data

class Dashboard:
    
    def __init__(self, title):
        st.set_page_config(
            page_title=title,
            layout="wide",
        )    
        self.build_sidebar()
        
        
    def build_sidebar(self):
        self.symbol = st.sidebar.selectbox('Symbol', SYMBOLS)
        self.timeframe = st.sidebar.selectbox('Timeframe', TIMEFRAMES)
        self.length = st.sidebar.selectbox('Length', range(50, 500, 50))
        
        
    def run(self):
        if self.symbol != "":
            data = self.get_price(self.symbol, self.timeframe, self.length)
            fig = self.create_fig(data)
            st.bokeh_chart(fig)

    def get_price(self, symbol, timeframe, length):
        mt5 = Mt5Manager(symbol, timeframe)
        return mt5.init(length, calc_indicators, {})

    def create_fig(self, data):
        time = data[Columns.JST]
        op = data[Columns.OPEN]
        hi = data[Columns.HIGH]
        lo = data[Columns.LOW]
        cl = data[Columns.CLOSE]
        chart = CandleChart(f'{self.symbol} {self.timeframe}', 1000, 400, time, op, hi, lo, cl)
        return chart.fig
                
def test():
    dashboard = Dashboard('NIKKEI')
    
    dashboard.run()
        
if __name__ == '__main__':
    test()