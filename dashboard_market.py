import time
import threading
import numpy as np
import pandas as pd
from dateutil import tz
from datetime import datetime, timedelta, timezone
import threading
import streamlit as st
from bokeh.layouts import column, row, layout, gridplot
import warnings
warnings.simplefilter('ignore')

from mt5_trade import Mt5Trade, Columns, PositionInfo
from time_utils import TimeUtils
from data_buffer import DataBuffer
from candle_chart import CandleChart, TimeChart
from technical import rally, SUPERTREND, SUPERTREND_SIGNAL, SQUEEZER, ANKO, ATRP
from common import Indicators

JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')  


from streamlit_autorefresh import st_autorefresh

# 5秒ごとにリロード
is_first_time = True

SYMBOLS = ['NIKKEI', 'DOW', 'NSDQ', 'SP', 'HK50', 'XAUUSD', 'USDJPY']
TIMEFRAMES = ['H1', 'M15', 'M30' 'H4', 'D1']
DATA_LENGTH = [300, 100, 200, 400, 500, 800, 1000]
UPDATE_COUNT = [5, 7, 10, 20]
DROP_RATE = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
LENGTH_MARGIN = 400

def calc_indicators(timeframe, data, technical_params):
    p = technical_params
    rally(data, long_term= p['ma_long_period'])
    SUPERTREND(data, p['atr_period'], p['atr_multiply'])
    ANKO(data, p['update_count'], p['profit_ma_period'], p['profit_target'], p['profit_drop'])
    w = p['atrp_period']
    ATRP(data, w, w)
    

    
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
        out = []
        if self.conditions is None:
            return out
        for symbol, timeframe, indicator_function, param, length, length_margin in self.conditions:
            df = self.mt5.get_rates(symbol, timeframe, length + length_margin)
            if remove_last:
                df = df.iloc[:-1, :]
            buffer = DataBuffer(symbol, timeframe, length, df, indicator_function, param, self.server_time_delta())
            out.append([symbol, timeframe, buffer.data])
            del df
        return out
    
    def run(self):
        self.loop = True
        while self.loop:
            self.data = self.get_price()
            time.sleep(0.5)
        
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
        self.set_param()
        self.placeholder = st.empty()
        self.loop = False
        self.mt5 =  Mt5Trade()
        self.mt5.connect()
        
    def set_param(self):
        self.symbol = st.sidebar.selectbox('Symbol', SYMBOLS)
        self.timeframe = st.sidebar.selectbox('Timeframe', TIMEFRAMES)
        self.length = st.sidebar.selectbox('Length', DATA_LENGTH)
        self.ma_long_period = st.sidebar.selectbox('MA Long period', [40, 20, 60, 100])
        self.atr_period = st.sidebar.selectbox('ATR period', [10, 15, 20, 25, 30])
        self.atr_multiply = st.sidebar.selectbox('ATR multiply', [3.0, 1.0, 1.5, 2.5, 3.0, 3.5, 4.0])
        self.update_count = st.sidebar.selectbox('Update count', UPDATE_COUNT)
        self.profit_ma_period = st.sidebar.selectbox('Profit MA period', [5, 3, 7, 10, 20])
        
        self.profit_target = [50, 100, 200, 300, 400]
        self.profit_drop   = [25, 50,   50, 100, 200]
        
        
    def separate(sellf, array, threshold):
        n = len(array)
        up = np.full(n, np.nan)
        down = np.full(n, np.nan)
        for i, a in enumerate(array):
            if a >= threshold:
                up[i] = a
            elif a <= -threshold:
                down[i] = a 
        return up, down
    
    
    def calc_range(self, lo, hi, k=2):
        vmin = np.nanmin(lo)
        vmax = np.nanmax(hi)
        r = (vmax - vmin) * 1.2
        n = np.round(np.log10(r), 0) 
        rate = 10 ** (n - 1)  
        r2 = r / rate  
        r3 = float(int(r2 + 0.5)) * rate
        return r3
        
    def expand(self, time, time1h, atr1h):
        n = len(time)
        atr = np.full(n, np.nan)
        for t1h, a in zip(time1h, atr1h):
            for i, t in enumerate(time):
                if t == t1h:
                    atr[i] = a
                    break
        for i in range(1, n):
            if np.isnan(atr[i]):
                if not np.isnan(atr[i - 1]):
                    atr[i] = atr[i - 1]                
        return atr
        
    def create_fig(self, data, data2):
        time = data[Columns.JST]
        op = data[Columns.OPEN]
        hi = data[Columns.HIGH]
        lo = data[Columns.LOW]
        cl = data[Columns.CLOSE]
        chart1 = CandleChart(f'{self.symbol} {self.timeframe}', None, 400, time)
        try:
            rally = data[Indicators.RALLY]
            chart1.plot_background(rally, ['green', 'red'])
        except:
            pass
        
        chart1.plot_candle(op, hi, lo, cl)
        #chart1.line(data[Indicators.MA_SHORT], color='red', alpha=0.5)
        #chart1.line(data[Indicators.MA_MID], color='green', alpha=0.5)
        chart1.line(data[Indicators.MA_LONG], color='blue', alpha=0.4, line_width=3.0, legend_label='MA Long')
        #chart1.line(data[Indicators.ATR_UPPER], color='orange', alpha=0.4, line_width=1)
        #chart1.line(data[Indicators.ATR_LOWER], color='green', alpha=0.4,No line_width=1)
        #chart1.line(data[Indicators.SUPERTREND_U], color='orange', alpha=0.4, line_width=4.0)
        #chart1.line(data[Indicators.SUPERTREND_L], color='green', alpha=0.4, line_width=4.0)
        #chart1.markers(data[Indicators.SQUEEZER], cl, 1, marker='o', color='red', alpha=0.5, size=10)
        chart1.set_ylim(np.min(lo), np.max(hi), self.calc_range(lo, hi))
        
        entry = data[Indicators.ANKO_ENTRY]
        ext = data[Indicators.ANKO_EXIT]
        chart1.markers(entry, cl, 1, marker='^', color='green', alpha=0.5, size=20)
        chart1.markers(entry, cl, -1, marker='v', color='red', alpha=0.5, size=20)
        chart1.markers(ext, cl, 1, marker='*', color='gray', alpha=0.6, size=30)
        chart1.markers(ext, cl, -1, marker='*', color='red', alpha=0.6, size=30)
        r = 50
        if self.timeframe == 'H1' or self.timeframe == 'M30':
            r = 100
        chart1.add_axis(yrange=[-r, r])
        update = data[Indicators.SUPERTREND_UPDATE]
        up_line, down_line = self.separate(update, self.update_count)
        chart1.line(update, extra_axis=True, color='yellow', alpha=0.5, line_width=5.0, line_dash='dotted')
        chart1.line(up_line, extra_axis=True, color='green', alpha=0.5, line_width=5.0, legend_label='Supertrend Count (Long)')
        chart1.line(down_line, extra_axis=True, color='red', alpha=0.5, line_width=5.0, legend_label='Supertrend Count (Short)')
        chart1.hline(0.0, 'black', extra_axis=True)
        chart1.fig.legend.location = 'top_left'
        
        chart2 = TimeChart('Profit', None, 200, time)
        chart2.line(data[Indicators.PROFITS],  color='blue', alpha=0.5, line_width=3.0, legend_label='Profit')
        chart2.line(data[Indicators.PROFITS_MA],  color='red', alpha=0.5, line_width=3.0, legend_label='Profit MA')
        chart2.markers(data[Indicators.PROFITS_PEAKS], data[Indicators.PROFITS], 1, marker='o', color='red', alpha=0.5, size=10)
        chart2.hline(0.0, 'black')
        chart2.fig.legend.location = 'top_left'
        chart2.fig.x_range = chart1.fig.x_range
        
        atrp1h = self.expand(time, data2[Columns.JST], data2[Indicators.ATRP])
        chart3 = TimeChart('ATRP', None, 200, time)
        chart3.line(data[Indicators.ATRP],  color='blue', alpha=0.5, line_width=3.0, legend_label='ATRP')
        chart3.line(atrp1h,  color='red', alpha=0.5, line_width=3.0, legend_label='ATRP H1')
        chart3.hline(0.0, 'black')
        chart3.fig.legend.location = 'top_left'
        chart3.fig.x_range = chart1.fig.x_range
        
        figs = [chart1.fig, chart2.fig, chart3.fig]
        l = column(*figs, sizing_mode='stretch_width', height=800, background='gray')
        return l #chart1.fig
    
                            
    def pickup_data(self, data, symbol, timeframe):
        for symb, tf, d in data:
            if symb == symbol and tf == timeframe:
                return d
        return None
                            
    def run(self):
        global is_first_time
        is_first_time = True
        if self.symbol != "":
            if not "DataLoader" in st.session_state:
                st.session_state.DataLoader = DataLoader(self.mt5)
                st.session_state.DataLoader.start()
            param = { 
                        'ma_long_period': self.ma_long_period,
                        'atr_period': self.atr_period,
                        'atr_multiply': self.atr_multiply,
                        'update_count': self.update_count,
                        'atrp_period': 40,
                        'profit_ma_period': self.profit_ma_period,
                        'profit_target': self.profit_target,
                        'profit_drop': self.profit_drop}
            
            c = [[self.symbol, self.timeframe, calc_indicators, param, self.length, LENGTH_MARGIN]]
            if self.timeframe != 'H1':
                c.append([self.symbol, 'H1', calc_indicators, {'atr_window': 40}, self.length, LENGTH_MARGIN])            
            st.session_state.DataLoader.setup(c)    
            self.loop = True    
        while self.loop:            
            d = st.session_state.DataLoader.data
            try:
                data = self.pickup_data(d, self.symbol, self.timeframe)
                data_atr = self.pickup_data(d, self.symbol, 'H1')
                container = self.create_fig(data, data_atr)
                if container is not None:
                    self.placeholder.bokeh_chart(container)
            except Exception as e:
                print(e)
            time.sleep(1)
            #if is_first_time:
            #    st_autorefresh(interval=5000, key="fizzbuzzcounter")
            #    is_first_time = False
 
def test():
    dashboard = Dashboard('NIKKEI')
    dashboard.run()
        
if __name__ == '__main__':
    test()