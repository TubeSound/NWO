import os
import shutil
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from random import randint, random
import math
import numpy as np
from candle_chart import TimeChart, CandleChart
from bokeh.layouts import column, row, layout, gridplot
from bokeh.io import export_png
from bokeh.plotting import show
from bokeh.plotting import output_notebook
output_notebook() 


import pandas as pd
import pickle
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

from common import Indicators, Columns, Signal
from technical import ANKO, rally, sma
from strategy import Simulation
from time_utils import TimeFilter, TimeUtils
from data_loader import DataLoader
import random
from dashboard_market import calc_indicators, calc_atrp, expand_time, calc_profit, PARAM, DEFAULT

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

def rand_step(begin, end, step):
    l = end - begin
    n = int(l / step + 0.5) + 1
    while True:
        r = randint(0, n)
        v = begin + r * step
        if v <= end:
            return v

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

def sl_fix(symbol):    
    if symbol == 'XAUUSD':
        sl = 10
    elif symbol == 'NIKKEI':
        sl = 500
    elif symbol == 'DOW':
        sl = 200
    elif symbol == 'NSDQ':
        sl = 50
    elif symbol == 'USDJPY':
        sl = 0.5
    return Simulation.SL_FIX, [sl]
            
def sl_range(symbol):
    if symbol == 'XAUUSD':
        sl = 10
    elif symbol == 'NIKKEI':
        sl = 200
    elif symbol == 'DOW':
        sl = 200
    elif symbol == 'NSDQ':
        sl = 50
    elif symbol == 'USDJPY':
        sl = 0.5        
    return Simulation.SL_RANGE, [10, sl] 
                
def get_trade_param(symbol, timeframe):
    try:
        p = PARAM[symbol][timeframe]
    except:
        p = {'sl': 500}
    param =  {
                'strategy': 'anko',
                'begin_hour': 0,
                'begin_minute': 0,
                'hours': 0,
                'sl_method': Simulation.SL_FIX,
                'sl_value': [p['sl']],
                'trail_target':0,
                'trail_stop': 0, 
                'volume': 0.1, 
                'position_max':2, 
                'timelimit': 0}
    return param


def randomize_trade_param(symbol):
    if symbol == 'NIKKEI' or symbol == 'DOW':
        begin = 50
        end = 500
        step = 25
    elif symbol == 'XAUUSD':
        begin = 5
        end = 100
        step = 5
    elif symbol == 'USDJPY' or symbol == 'GBPJPY':
        begin = 0.2
        end = 2.0
        step = 0.2
    elif symbol == 'NSDQ':
        begin = 10
        end = 100
        step = 10
    elif symbol == 'CL':
        begin = 0.1
        end = 5
        step = 0.1

    param =  {
                'strategy': 'anko',
                'begin_hour': 0,
                'begin_minute': 0,
                'hours': 0,
                'sl_method': Simulation.SL_FIX,
                'sl_value': [rand_step(begin, end, step)],
                'trail_target':0,
                'trail_stop': 0, 
                'volume': 0.1, 
                'position_max':2, 
                'timelimit': 0}
    return param

    
def get_technical_param(symbol, timeframe):
    
    try:
        p = PARAM[symbol][timeframe]
    except:
        p = DEFAULT
    param = { 
            'ma_long_period': p['ma_long_period'],
            'atr_period': p['atr_period'],
            'atr_multiply': p['atr_multiply'],
            'update_count': p['update_count'],
            'atrp_period':40,
            'profit_ma_period': 5,
            'profit_target': [50, 100, 200, 300, 400],
            'profit_drop':  [25, 50,   50, 100, 200]
            }
    return param        

def randomize_technical_param():
    param = { 
            'ma_long_period': rand_step(20, 80, 5),
            'atr_period': rand_step(5, 20, 5),
            'atr_multiply': rand_step(1, 4, 0.5),
            'update_count': rand_step(1, 30, 1),
            'atrp_period': 40,
            'profit_ma_period': 5,
            'profit_target': [50, 100, 200, 300, 400],
            'profit_drop':  [25, 50,   50, 100, 200]
            }
    return param           
  
# -----
              
class Backtest():
    
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = self.from_pickle(symbol, timeframe)
        if timeframe != 'H1':
            self.data_h1 = self.from_pickle(symbol, 'H1')
        else:
            self.data_h1 = None
        self.jst = self.data[Columns.JST]
                
    def from_pickle(self, symbol, timeframe):
        filepath = './data/Axiory/' + symbol + '_' + timeframe + '.pkl'
        with open(filepath, 'rb') as f:
            data0 = pickle.load(f)
        return data0                
    
    def set_time(self, tbegin, tend):
        if tend is None:
            tend = self.jst[-1]
        n, data = TimeUtils.slice(self.data, Columns.JST, tbegin, tend)   
        if n > 0:
            self.data = data
            self.jst = self.data[Columns.JST]
    
    
    def calc_drawdown(self, profit_data, window=5):
        def search_upper(data, ibegin, value):
            n = len(data)
            for i in range(ibegin, n): 
                if data[i] > value:
                    return i
            return -1     

        def calc_down(data, index, j):
            d = data[index: j]
            imin = np.argmin(d)
            return imin + index
            
        time = profit_data[0]
        profits = profit_data[1]
        ma = sma(profits, window)
        n = len(time)
        drawdowns = []
        sum_drawdown = 0
        begin_value = None
        ibegin = None
        i = window
        while i <  n - 1:
            if ma[i] < ma[i - 1]:
                begin_value = ma[i]
                ibegin = i
                iend = search_upper(ma, ibegin, begin_value)
                if iend >= 0:
                    ilow = calc_down(ma, ibegin, iend)
                    drawdowns.append([ibegin, begin_value, ilow, ma[ilow], iend, ma[iend]])
                    sum_drawdown += (ma[ilow] - begin_value)
                    i = iend + 1
                else:
                    ilow = calc_down(ma, ibegin, n - 1)
                    drawdowns.append([ibegin, begin_value, ilow, ma[ilow], n - 1, ma[-1]])
                    sum_drawdown += (ma[ilow] - begin_value)
                    break
            i += 1
        return drawdowns, sum_drawdown
        

    def trade(self, data, trade_param):
        sim = Simulation(data, trade_param)        
        ret = sim.run(data, Indicators.ANKO_ENTRY, Indicators.ANKO_EXIT)
        df, summary, profit_curve = ret
        _, drawdown = self.calc_drawdown(profit_curve)
        trade_num, profit, win_rate = summary
        return (df, summary, drawdown, profit_curve)
        
    def evaluate(self, technical_param, trade_param, dirpath, save=True, plot=True):
        calc_indicators(self.data, technical_param)
        if self.data_h1 is not None:
            calc_atrp(self.data_h1, technical_param)
            atr_h1 = self.data_h1[Indicators.ATR]
            atr_h1 = expand_time(self.data[Columns.JST], self.data_h1[Columns.JST], atr_h1)
            atrp_h1 = self.data_h1[Indicators.ATRP]
            atrp_h1 = expand_time(self.data[Columns.JST], self.data_h1[Columns.JST], atrp_h1)
        else:
            atr_h1 = self.data[Indicators.ATR]
            atrp_h1 = self.data[Indicators.ATRP]
        self.data[Indicators.ATR_H1] = atr_h1
        self.data[Indicators.ATRP_H1] = atrp_h1
        (df, summary, drawdown, profit_curve) = self.trade(self.data, trade_param)
        trade_num, profit, win_rate = summary
        #print(summary)
        df.to_csv(os.path.join(dirpath, 'trade_summary.csv'), index=False)
        if len(profit_curve[0]) < 2:
            return df, summary, drawdown, profit_curve, None
        title = f'[ANKO] {self.symbol} {self.timeframe} Total Profit: {profit} n: {trade_num} win: {win_rate * 100}%'
        chart = TimeChart(title, 800, 400, profit_curve[0])
        chart.line(profit_curve[1], color='blue')
        if save:
            chart.to_png(os.path.join(dirpath, 'profit.png'))    
        if plot:
            show(chart.fig)
        return (df, summary, drawdown, profit_curve, chart) 
    
    
    
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
    loscut = df['losscuted'].to_list()
    for s, t0, t1, p0, p1, lc in zip(signal, t_entry, t_exit, price_entry, price_exit, loscut):
        if t1 >= time[0] and t1 <= time[-1]:
            if s == 1:
                marker = '^'
                color = 'green'
            elif s == -1:
                marker = 'v'
                color = 'red'
            fig.marker(t0, p0, marker=marker, color=color, alpha=0.5, size=20)
            marker = 'x' if lc == 'true' else '*'
            fig.marker(t1, p1, marker=marker, color='gray', alpha=0.7, size=30)
            
def plot_entry_exit_markers(chart, profits, entries, exits):
    for i, (en, ex) in enumerate(zip(entries, exits)):
        v = 0 if np.isnan(profits[i]) else profits[i]
        if en == 1:
            color='red'
            chart.marker(i, v, marker='o', color=color, alpha=0.5, size=10)
        elif en == -1:
            color = 'green'
            chart.marker(i, v, marker='o', color=color, alpha=0.5, size=10)
        if ex == 1:
            chart.marker(i, v, marker='x', color='gray', alpha=0.5, size=20)            
        
def main(symbol, timeframe, save=True, plot=True):
    days = 20
    backtest = Backtest(symbol, timeframe)
    root = f'./evaluate/{symbol}/{timeframe}'
    os.makedirs(root, exist_ok=True)
    trade_param = get_trade_param(symbol, timeframe)
    technical_param = get_technical_param(symbol, timeframe)
    df = backtest.evaluate(technical_param, trade_param, root, plot=(not plot))
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
            l = create_fig(data, df) 
            if save:
                export_png(l, filename=os.path.join(dirpath, f'{count}.png'))  
            if plot:
                show(l)
            count += 1
        t0 = t1
        t1 = t0 + timedelta(days=days)
        n, data = TimeUtils.slice(data0, Columns.JST, t0, t1)


def optimize(symbol, timeframe, loop=200):
    root = f'./optimize/sl_fix/{symbol}/{timeframe}'
    os.makedirs(root, exist_ok=True)
    png_dir = os.path.join(root, 'profit')
    os.makedirs(png_dir, exist_ok=True)
    data = []
    for i in range(loop):
        backtest = Backtest(symbol, timeframe)
        trade_param = randomize_trade_param(symbol)
        technical_param = randomize_technical_param()
        (df, summary, drawdown, profit_curve, chart_profit) = backtest.evaluate(technical_param, trade_param, root, plot=False)
        trade_num, profit, win_rate = summary
        p = [i, trade_num, profit, drawdown, win_rate]
        print(i, profit)
        columns = ['no', 'num', 'profit', 'drawdown', 'win_rate']
        p0, c0 = expand('tech', technical_param) 
        p1, c1 = expand('trade', trade_param)
        p += p0
        p += p1
        data.append(p)
        columns += c0
        columns += c1
        try:
            df_summary = pd.DataFrame(data=data, columns=columns)
            df_summary.to_excel(os.path.join(root, f'summary.xlsx'))
        except:
            pass
        if profit > 30:
            chart_profit.to_png(os.path.join(png_dir, f'{i:04}_profit.png'))    


def select_top(array, index, top):
    n = len(array)
    if n <= top:
        return array
    else:
        return sorted(array, key=lambda x: x[index], reverse=True)[:top]
        
def optimize2stage(symbol, timeframe, repeat=1000, top=50):
    dirpath = f'./optimize2stage_2025_01/sl_fix/{symbol}/{timeframe}'
    os.makedirs(dirpath, exist_ok=True)
    png_dir = os.path.join(dirpath, 'profit')
    os.makedirs(png_dir, exist_ok=True)
    tbegin = datetime(2025, 1, 1).astimezone(JST)
    columns = None
    result = []
    for i in range(repeat):
        backtest = Backtest(symbol, timeframe)
        backtest.set_time(tbegin, None)
        trade_param = randomize_trade_param(symbol)
        technical_param = randomize_technical_param()
        (df, summary, drawdown, profit_curve, chart_profit) = backtest.evaluate(technical_param, trade_param, dirpath, plot=False)
        trade_num, profit, win_rate = summary
        p = [i, profit, drawdown, technical_param, trade_param]
        print(i, profit, drawdown)
        result.append(p)
    
    selected = select_top(result, 1, top) 
    
    data = []
    columns = None
    for i, (_,  profit, drawdown, technical_param, trade_param) in enumerate(selected):
        backtest = Backtest(symbol, timeframe)
        (df, summary, drawdown, profit_curve, chart_profit) = backtest.evaluate(technical_param, trade_param, dirpath, plot=False)
        trade_num, profit, win_rate = summary
        if profit > 0:
            chart_profit.to_png(os.path.join(png_dir, f'{i:04}_{symbol}_{timeframe}_profit.png'))    

        p = [i, trade_num, profit, drawdown, win_rate]
        p0, c0 = expand('', technical_param) 
        p1, c1 = expand('', trade_param)
        p += p0
        p += p1
        data.append(p)
        if columns is None:
            columns = ['no', 'num', 'profit', 'drawdown', 'win_rate']
            columns += c0
            columns += c1
    df_summary = pd.DataFrame(data=data, columns=columns)
    df_best = select_best_param(df_summary)
    df_summary.sort_values(by='profit', ascending=False)
    df_summary.to_csv(os.path.join(dirpath, f'{symbol}_{timeframe}_trade_param.csv'), index=False)
        
        
def select_best_param(df0):
    def rotate(point, center, angle):
        x = (point[0] - center[0]) * math.cos(angle) - (point[1] - center[1]) * math.sin(angle) + center[0]
        y = (point[1] - center[1]) * math.cos(angle) + (point[0] - center[0]) * math.sin(angle) + center[1]
        return (x, y)
    
    df = df0[df0['profit'] > 0]
    print(df.columns)
    no = df['no'].to_numpy()
    profit = df['profit'].to_numpy()
    drawdown = df['drawdown'].to_numpy()
    p0 = [min(profit), min(drawdown)]
    p1 = [max(profit), max(drawdown)]
    center = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2] 
    vector = np.array(p1) - np.array(p0)
    angle = np.arctan2(vector[0], vector[1])
    xs = []
    ys = []
    for p, d in zip(profit, drawdown):
        x, y = rotate((p, d), center, -angle)
        xs.append(x)
        ys.append(y)
    imax = np.argmax(xs)
    print(no[imax], profit[imax], drawdown[imax])
    return df[df['no'] == no[imax]]
     

   
def debug(symbol, timeframe, save=True, plot=True):
    days = 10
    backtest = Backtest(symbol, timeframe)
    root = f'./debug/{symbol}/{timeframe}'
    os.makedirs(root, exist_ok=True)
    data0 = backtest.data
    #analyze(data0)
    time = data0[Columns.JST]
    
    tbegin = time[0]
    tend = time[-1]
    
    t1 = tend
    t0 = datetime(2021, 5, 5).astimezone(JST)
    t1 = datetime(2021, 5, 14).astimezone(JST)
    n, data = TimeUtils.slice(data0, Columns.JST, t0, t1)
    backtest.data = data    
    trade_param = get_trade_param(symbol, timeframe)
    technical_param = get_technical_param(symbol, timeframe)
    df = backtest.evaluate(technical_param, trade_param, root)
    l = create_fig(data, df) 
    if save:
        export_png(l, filename=os.path.join(root, '0.png'))  
    if plot:
        show(l)
    
    dirpath = os.path.join(root, 'chart')
    os.makedirs(dirpath, exist_ok=True)   
    df2 = pd.DataFrame(data)
    if save:
        df2.to_csv(os.path.join(root, 'data.csv'))
    return l
     
def analyze(data):
     for key, value in data.items():
         print(key, type(value), len(value))    
         
def separate(array, threshold):
    n = len(array)
    up = np.full(n, np.nan)
    down = np.full(n, np.nan)
    for i, a in enumerate(array):
        if a >= threshold:
            up[i] = a
        elif a <= -threshold:
            down[i] = a 
    return up, down     
    
def calc_range(lo, hi, k=2):
    vmin = np.nanmin(lo)
    vmax = np.nanmax(hi)
    r = (vmax - vmin) * 1.2
    n = np.round(np.log10(r), 0) 
    rate = 10 ** (n - 1)  
    r2 = r / rate  
    r3 = float(int(r2 + 0.5)) * rate
    return r3
         
def create_fig(data, df, plot_marker=False):
    w = 1200
    time = data[Columns.JST]
    op = data[Columns.OPEN]
    hi = data[Columns.HIGH]
    lo = data[Columns.LOW]
    cl = data[Columns.CLOSE]
    profits = data[Indicators.PROFITS]
    trade_n, total_profit = calc_profit(profits)
    chart1 = CandleChart(f'trade n: {trade_n} profit: {total_profit:.3f}', w, 350, time)
    rally = data[Indicators.RALLY]
    chart1.plot_background(rally, ['green', 'red'])
    chart1.plot_candle(op, hi, lo, cl)
    #chart1.line(data[Indicators.MA_SHORT], color='red', alpha=0.5)
    #chart1.line(data[Indicators.MA_MID], color='green', alpha=0.5)
    chart1.line(data[Indicators.MA_LONG], color='blue', alpha=0.4, line_width=3.0, legend_label='MA Long')
    #chart1.line(data[Indicators.ATR_UPPER], color='orange', alpha=0.4, line_width=1)
    #chart1.line(data[Indicators.ATR_LOWER], color='green', alpha=0.4, line_width=1)
    chart1.line(data[Indicators.SUPERTREND_U], color='orange', alpha=0.4, line_width=4.0)
    chart1.line(data[Indicators.SUPERTREND_L], color='green', alpha=0.4, line_width=4.0)
    #chart1.markers(data[Indicators.SQUEEZER], cl, 1, marker='o', color='red', alpha=0.5, size=10)
    chart1.set_ylim(np.min(lo), np.max(hi), calc_range(lo, hi))
    
    r = 50
    chart1.add_axis(yrange=[-r, r])
    update = data[Indicators.SUPERTREND_UPDATE]
    up_line, down_line = separate(update, 5)
    chart1.line(update, extra_axis=True, color='yellow', alpha=0.5, line_width=5.0, line_dash='dotted')
    #chart1.line(up_line, extra_axis=True, color='green', alpha=0.5, line_width=5.0, legend_label='Supertrend Count (Long)')
    #chart1.line(down_line, extra_axis=True, color='red', alpha=0.5, line_width=5.0, legend_label='Supertrend Count (Short)')
    chart1.hline(0.0, 'black', extra_axis=True)
    chart1.fig.legend.location = 'top_left'
    if plot_marker:
        plot_markers(chart1, time, df)
    
    chart2 = TimeChart('Profit', w, 150, time)
    chart2.line(data[Indicators.PROFITS],  color='blue', alpha=0.5, line_width=3.0, legend_label='Profit')
    plot_entry_exit_markers(chart2, data[Indicators.PROFITS], data[Indicators.ANKO_ENTRY], data[Indicators.ANKO_EXIT])
    chart2.fig.legend.location = 'top_left'
    chart2.fig.x_range = chart1.fig.x_range
    
    chart3 = TimeChart('ATRP', w, 150, time)
    chart3.line(data[Indicators.ATRP],  color='blue', alpha=0.5, line_width=3.0, legend_label='ATRP')
    chart3.hline(0.0, 'black')
    chart3.fig.legend.location = 'top_left'
    chart3.fig.x_range = chart1.fig.x_range
    
    chart4 = TimeChart('ATRP', None, 150, time)
    chart4.line(data[Indicators.ATRP],  color='blue', alpha=0.5, line_width=3.0, legend_label='ATRP')
    chart4.line(data[Indicators.ATRP_H1],  color='red', alpha=0.5, line_width=3.0, legend_label='ATRP H1')
    chart4.hline(0.0, 'black')
    chart4.fig.legend.location = 'top_left'
    chart4.fig.x_range = chart1.fig.x_range
    chart4.set_ylim(0, 0.5, 0.5)
    chart4.hline(0.25, 'yellow', width=2.0)
    
        
    figs = [chart1.fig, chart2.fig, chart3.fig, chart4.fig]
    l = column(*figs, width=w, height=800, background='gray')
    return l #chart1.fig
    

    
    
    
if __name__ == '__main__':
    #optimize2stage('CL', 'H1')
    #main('DOW','H1')
    debug('NIKKEI', 'H1')