import time
import threading
import numpy as np
import pandas as pd
from dateutil import tz
from datetime import datetime, timedelta, timezone
import threading
import streamlit as st
import warnings

warnings.simplefilter('ignore')


from mt5_trade import Mt5Trade, Columns, PositionInfo
from time_utils import TimeUtils
from data_buffer import DataBuffer
from candle_chart import CandleChart, TimeChart
from technical import rally, SUPERTREND, SUPERTREND_SIGNAL, ANKO
from common import Indicators

JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')  

SYMBOLS = ['', 'NIKKEI', 'DOW', 'NSDQ']
TIMEFRAMES = ['M15', 'M30', 'H1', 'H4', 'D1']

LENGTH_MARGIN = 400

def calc_indicators(timeframe, data, technical_params):
    rally(data)
    SUPERTREND(data, 10, 3.0)
    SUPERTREND_SIGNAL(data, 10)
    ANKO(data)
    
class Mt5Manager:
    def __init__(self, mt5, symbol, timeframe):
        self.set_sever_time()
        self.mt5 = mt5
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
        #print('SeverTime GMT+', dt, tz)
        
    def init(self, length, length_margin, technical_function, technical_param, remove_last=False):
        df = self.mt5.get_rates(self.symbol, self.timeframe, length + length_margin)
        if remove_last:
            df = df.iloc[:-1, :]
        buffer = DataBuffer(technical_function, self.symbol, self.timeframe, length, df, technical_param, self.delta_hour_from_gmt)
        self.buffer = buffer
        return buffer.data

class DataLoader(threading.Thread):
    def __init__(self, mt5, **kwargs):
        super().__init__(**kwargs)
        self.data = None
        self.mt5 = mt5
        self.conditions = None
    
    def get_price(self, remove_last=False):
        out = {}
        if self.conditions is None:
            return out
        for symbol, dic in self.conditions.items():
            d = {}
            for timeframe, [indicator_function, param, length, length_margin] in dic.items():
                df = self.mt5.get_rates(symbol, timeframe, length + length_margin)
                if remove_last:
                    df = df.iloc[:-1, :]
                buffer = DataBuffer(symbol, timeframe, length, df, indicator_function, param, self.server_time_delta())
                d[timeframe] = buffer.data
                del df
            out[symbol] = d
        return out
    
    def run(self):
        self.loop = True
        while self.loop:
            self.data = self.get_price()
            time.sleep(10)
        
    def setup(self, conditions):
        self.conditions = conditions
        
    def server_time_delta(self):
        begin_month = 3
        begin_sunday = 2
        end_month = 11
        end_sunday = 1
        delta_hour_from_gmt_in_summer = 3
        now = datetime.now(JST)
        dt, tz = TimeUtils.delta_hour_from_gmt(now, begin_month, begin_sunday, end_month, end_sunday, delta_hour_from_gmt_in_summer)
        self.delta_hour_from_gmt  = dt
        self.server_timezone = tz
        return dt        

class Dashboard:
    def __init__(self, title):
        st.set_page_config(
            page_title=title,
            layout="wide",
        )    
        self.build_sidebar()
        self.placeholder = st.empty()
        self.loop = False
        self.mt5 =  Mt5Trade()
        self.mt5.connect()
        
    def build_sidebar(self):
        self.symbol = st.sidebar.selectbox('Symbol', SYMBOLS)
        self.timeframe = st.sidebar.selectbox('Timeframe', TIMEFRAMES)
        self.length = st.sidebar.selectbox('Length', range(100, 500, 50))
        
    def create_fig(self, data):
        time = data[Columns.JST]
        op = data[Columns.OPEN]
        hi = data[Columns.HIGH]
        lo = data[Columns.LOW]
        cl = data[Columns.CLOSE]
        chart1 = CandleChart(f'{self.symbol} {self.timeframe}', 1000, 600, time)
        chart1.plot_background(data[Indicators.RALLY], ['green', 'red'])
        chart1.plot_candle(op, hi, lo, cl)
        chart1.line(data[Indicators.MA_SHORT], color='red', alpha=0.5)
        chart1.line(data[Indicators.MA_MID], color='green', alpha=0.5)
        chart1.line(data[Indicators.MA_LONG], color='blue', alpha=0.5)
        chart1.line(data[Indicators.SUPERTREND_U], color='green', alpha=0.4, line_width=4.0)
        chart1.line(data[Indicators.SUPERTREND_L], color='orange', alpha=0.4, line_width=4.0)
        
        entry = data[Indicators.ANKO_ENTRY]
        ext = data[Indicators.ANKO_EXIT]
        chart1.markers(entry, cl, 1, marker='^', color='green', alpha=0.7, size=20)
        chart1.markers(entry, cl, -1, marker='v', color='red', alpha=0.7, size=20)
        chart1.markers(ext, cl, 1, marker='x', color='black', alpha=0.7, size=30)
        chart2 = TimeChart('', 1000, 200, time)
        chart2.line(data[Indicators.RALLY],  color='blue', alpha=0.5, line_width=3.0)
        chart2.line(data[Indicators.SUPERTREND], color='red', alpha=0.5, line_width=3.0)
        return [chart1.fig, chart2.fig]
    
                            
    def run(self):
        if self.symbol != "":
            if not "DataLoader" in st.session_state:
                st.session_state.DataLoader = DataLoader(self.mt5)
                st.session_state.DataLoader.start()
            conditions = {self.symbol: {self.timeframe: [calc_indicators, {}, self.length, LENGTH_MARGIN]}}
            st.session_state.DataLoader.setup(conditions)    
            self.loop = True    
        while self.loop:            
            dic = st.session_state.DataLoader.data
            try:
                data = dic[self.symbol][self.timeframe]
                figs = self.create_fig(data)
                container = st.container()
                container.bokeh_chart(figs[0])
                container.bokeh_chart(figs[1])
                self.placeholder.bokeh_chart(container)
            except Exception as e:
                print(e)
            time.sleep(10)
 
def test():
    dashboard = Dashboard('NIKKEI')
    dashboard.run()
        
if __name__ == '__main__':
    test()