import os
import shutil
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import random
import math
import numpy as np
from candle_chart import TimeChart, CandleChart
from bokeh.layouts import column, row, layout, gridplot
from bokeh.io import export_png

import pandas as pd
import pickle
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

from common import Indicators, Columns, Signal
from technical import SUPERTREND, SUPERTREND_SIGNAL
from strategy import Simulation
from time_utils import TimeFilter, TimeUtils
from data_loader import DataLoader
import random

def makeFig(rows, cols, size):
    fig, ax = plt.subplots(rows, cols, figsize=(size[0], size[1]))
    return (fig, ax)

def gridFig(row_rate, size):
    rows = sum(row_rate)
    fig = plt.figure(figsize=size)
    gs = gridspec.GridSpec(rows, 1, hspace=0.6)
    axes = []
    begin = 0
    for rate in row_rate:
        end = begin + rate
        ax = plt.subplot(gs[begin: end, 0])
        axes.append(ax)
        begin = end
    return (fig, axes)

def expand(name: str, dic: dict):
    data = []
    columns = []
    for key, value in dic.items():
        if name == '':
            column = key
        else:
            column = name + '.' + key
        if type(value) == dict:
            d, c = expand(column, value)                    
            data += d
            columns += c
        else:
            data.append(value)
            columns.append(column)
    return data, columns 



def timefilter(data, year_from, month_from, day_from, year_to, month_to, day_to):
    t0 = datetime(year_from, month_from, day_from).astimezone(JST)
    t1 = datetime(year_to, month_to, day_to).astimezone(JST)
    return TimeUtils.slice(data, data['jst'], t0, t1)


def plot_marker(ax, data, signal, markers, colors, alpha=0.5, s=50):
    time = data[Columns.JST]
    cl = data[Columns.CLOSE]
    for i, status in enumerate(signal):
        if status == 1:
            color = colors[0]
            marker = markers[0]
        elif status == -1:
            color = colors[1]
            marker = markers[1]
        else:
            continue
        ax.scatter(time[i], cl[i], color=color, marker=marker, alpha=alpha, s=s)
  
def plot_profit(ax, df, t0, t1, rng):
    df['texit'] = pd.to_datetime(df['exit_time'])
    df['tentry'] = pd.to_datetime(df['entry_time'])
    df2 = df[df['texit'] >= t0]
    df2 = df2[df2['texit'] <= t1]
    print('trade count: ', len(df2))
    
    signal = df2['signal'].to_list()
    tentry = df2['tentry'].to_list()
    price1 = df2['entry_price'].to_list()
    texit = df2['texit'].to_list()
    price2 = df2['exit_price'].to_list()
    profits = df2['profit'].to_list()
    for sig, ten, tex, p1, p2, prof in zip(signal, tentry, texit, price1, price2, profits):
        if sig == 1:
            color='green'
        elif sig == -1:
            color='red'
        else:
            continue
        ax.vlines(ten, rng[0], rng[1], color=color)
        ax.vlines(tex, rng[0], rng[1], color='gray')
        if prof > 0:
            ax.text(tex, rng[1], f'{prof:.3f}', color='green')
        else:
            ax.text(tex, rng[0], f'{prof:.3f}', color='red')
              
class Backtest():
    
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = self.from_pickle(symbol, timeframe)
                
    def from_pickle(self, symbol, timeframe):
        filepath = '../Blackswan/data/Axiory/' + symbol + '_' + timeframe + '.pkl'
        with open(filepath, 'rb') as f:
            data0 = pickle.load(f)
        return data0                
                
    def make_trade_param(self, symbol, randomize=False):
        begin_hour = 0
        begin_minute = 0
        hours = 0
        if symbol == 'XAUUSD':
            k = 0.1
        if symbol == 'XAGUSD':
            k = 0.001
        elif symbol == 'XPDUSD':
            k = 0.005
        elif symbol == 'CL':
            k = 0.002
        elif symbol == 'NSDQ':
            k = 0.5
        elif symbol == 'USDJPY':
            k = 0.001
        elif symbol == 'TSLA':
            k = 0.01
            begin_hour = 21
            begin_minute = 0
            hours = 7
        elif symbol == 'NVDA':
            k = 0.003
            begin_hour = 21
            begin_minute = 0
            hours = 7
        elif symbol == 'DAX':
            k = 0.5
        elif symbol == 'HK50':
            k = 0.5
        else:
            k = 1.0
        
        if randomize:
            sl = random.randint(1, 10) * 20
            trail_target = random.randint(1, 10) * 50
            trail_stop = random.randint(1, 10) * 20
        else:
            sl = 250
            trail_target = 300
            trail_stop = 200
        
        param =  {
                    'strategy': 'supertrend',
                    'begin_hour': begin_hour,
                    'begin_minute': begin_minute,
                    'hours': hours,
                    'sl_method': Simulation.SL_FIX,
                    'sl_value': int(sl * k),
                    'trail_target': int(trail_target * k),
                    'trail_stop': int(trail_stop * k), 
                    'volume': 0.1, 
                    'position_max':2, 
                    'timelimit': 0}
        return param, k        

    def trade(self, data, trade_param):
        sim = Simulation(data, trade_param)        
        ret = sim.run(data,
                                            Indicators.SUPERTREND_ENTRY,
                                            Indicators.SUPERTREND_EXIT,
                                            Indicators.ATR_UPPER,
                                            Indicators.ATR_LOWER)
        df, summary, profit_curve = ret
        trade_num, profit, win_rate = summary
        return (df, summary, profit_curve)
        
    def evaluate(self, dirpath):
        trade_param, k = self.make_trade_param(self.symbol)
        SUPERTREND(self.data, 10, 3.0)
        SUPERTREND_SIGNAL(self.data)
        (df, summary, profit_curve) = self.trade(self.data, trade_param)
        print(summary)
        df.to_csv(os.path.join(dirpath, 'trade_summary.csv'), index=False)
        
        chart = TimeChart('NIKKEI M30 SUPERTREND', 800, 400, profit_curve[0])
        chart.line(profit_curve[1], color='blue')
        chart.to_png(os.path.join(dirpath, 'profit.png'))    
        return df
    
    
    
def plot_markers(fig, time, df):
    def to_datetime(df, column):
        out = []
        for s in df[column]:
            t = datetime.strptime(s[:19], '%Y-%m-%d %H:%M:%S').astimezone(JST)
            out.append(t)
        return out
    signal = df['signal'].to_list()
    t_entry = to_datetime(df, 'entry_time')
    t_exit = to_datetime(df, 'exit_time')
    price_entry = df['entry_price'].to_list()
    price_exit = df['exit_price'].to_list()
    for s, t0, t1, p0, p1 in zip(signal, t_entry, t_exit, price_entry, price_exit):
        if t1 >= time[0] and t1 <= time[-1]:
            if s == 1:
                marker = '^'
                color = 'green'
            elif s == -1:
                marker = 'v'
                color = 'red'
            fig.marker(t0, p0, marker=marker, color=color, alpha=0.5, size=20)
            fig.marker(t1, p1, marker='x', color='gray', alpha=0.5, size=20)
    
        
def main():
    days = 10
    symbol = 'NIKKEI'
    timeframe = 'M30'
    backtest = Backtest(symbol, timeframe)
    root = f'./evaluate/{symbol}/{timeframe}'
    os.makedirs(root, exist_ok=True)
    df = backtest.evaluate(root)
    
    data0 = backtest.data
    time = data0[Columns.JST]
    
    tbegin = time[0]
    tend = time[-1]
    
    t0 = tbegin
    t1 = tbegin + timedelta(days=days)
    n, data = TimeUtils.slice(data0, Columns.JST, t0, t1)
    dirpath = os.path.join(root, 'chart')
    os.makedirs(dirpath, exist_ok=True)
    count = 1
    while t1 <= tend:
        if n > 50:
            jst = data[Columns.JST]
            op = data[Columns.OPEN]
            hi = data[Columns.HIGH]
            lo = data[Columns.LOW]
            cl = data[Columns.CLOSE]
            chart1 = CandleChart(f'{symbol} {timeframe} Supertrend', 1200, 500, jst)
            chart1.plot_candle(op, hi, lo, cl)   
           
            upper = data[Indicators.SUPERTREND_U]
            lower = data[Indicators.SUPERTREND_L]
            chart1.line(upper, color='green')
            chart1.line(lower, color='orange')
            plot_markers(chart1, jst, df)
            chart2 = TimeChart('Update', 1200, 200, jst)
            chart2.line(data[Indicators.SUPERTREND_UPDATE], color='red')
            figs = [chart1.fig, chart2.fig]
            l = column(*figs, width=1220, height=720, background='gray')
            export_png(l, filename=os.path.join(dirpath, f'{count}.png'))  
            count += 1
        t0 = t1
        t1 = t0 + timedelta(days=days)
        n, data = TimeUtils.slice(data0, Columns.JST, t0, t1)
    
if __name__ == '__main__':
    main()