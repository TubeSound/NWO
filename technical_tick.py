import numpy as np 
import pandas as pd
import math
from datetime import datetime, timedelta
from dateutil import tz
from technical import sma, linear_regression, is_nans
import os
from dateutil import tz
import matplotlib.pyplot as plt
from candle_chart import TimeChart



JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc') 

    
def time_diff(time):
    n = len(time)
    out = np.full(n, np.nan)
    for i in range(1, n):
        diff = time[i] - time[i - 1]
        dt  = diff.seconds + diff.microseconds / 1e6
        if dt == 0.0:
            dt = 0.1
        out[i] = dt
    return np.array(out)    
    
def change_percent(prices):
     change = (np.diff(prices) / prices[:-1])
     return insert_nan(change)

def insert_nan(array):
    return np.insert(array, 0, np.nan)

    
def detect_drop_spike(df, time, prices, jerk_ticks=100, jerk_drop_th=2.0, jerk_spike_th=1.5, acc_th=0.02):
    # 時間差（秒）
    tdiff = time_diff(time)
    change = change_percent(prices)
    velocity = change / tdiff
    acc = insert_nan(np.diff(velocity)) / tdiff
    jerk = insert_nan(np.diff(acc))/ tdiff
    jerk_energy = jerk ** 2
    jerk_energy_ma = np.array(sma(jerk_energy, jerk_ticks))
    drop = np.full(len(time), np.nan)
    spike = np.full(len(time), np.nan)
    drop_time= []
    drop_value = []
    spike_time = []
    spike_value = []
    for i, (e, a) in enumerate(zip(jerk_energy_ma, acc)):
        if a < - 1 * acc_th and e >= jerk_drop_th:
            drop[i] = e
            drop_time.append(time[i])
            drop_value.append(prices[i])        
        if a > acc_th and e >= jerk_spike_th:
            spike[i] = e
            spike_time.append(time[i])
            spike_value.append(prices[i])    
    df['pct_change'] = change
    df['velocity'] = velocity
    df['acc'] = acc
    df['jerk'] = jerk
    df['jerk_energy'] = jerk_energy
    df['jerk_energy_ma'] = jerk_energy_ma
    return (drop, drop_time, drop_value), (spike, spike_time, spike_value)


def slope(tdelta, price):
    s = 0.0
    time = []
    for t in tdelta:
        s += t
        time.append(s)
    values = (price - price[0])/price[0] * 60 * 100
    slope, offset = linear_regression(time, values)
    return slope
    
def rebound(df, time, prices, ticks=100):
    n = len(prices)
    tdiff = time_diff(time)
    rebound_ratios = np.full(n, np.nan)
    change_rates = np.full(n, np.nan)
    slopes = np.full(n, np.nan)
    for i in range(ticks - 1, n):
        vector = prices[i - ticks + 1: i + 1]
        tdelta = tdiff[i - ticks + 1: i + 1]
        if is_nans(vector):
            continue
        slp = slope(tdelta, vector)
        slopes[i] = slp
        if vector[0] == 0.0:
            continue
        # 下落率を計算
        change_rate = (vector[-1] - vector[0]) / vector[0]
        change_rates[i] = change_rate
        # 価格差分（Tickごとの差）を求める
        diffs = np.diff(vector)
        # 正（上昇）のTickの数を数える
        if slp >= 0:
            rebound_count = np.sum(diffs <= 0)    
        else:
            rebound_count = np.sum(diffs > 0)
        # 全体に対する反発Tickの割合（0〜1の範囲）
        rebound_ratio = rebound_count / len(diffs)
        #if rebound_ratio < rebound_ratio_th:
        rebound_ratios[i] = rebound_ratio
    df['rebound_ratio'] = rebound_ratios
    df['change_rate'] = change_rates
    df['slope'] = slopes
    
    
def detect_bottom(time, prices, points, term_sec=60):
    base = time[0]
    seconds = np.array([(t - base).total_seconds() for t in time])
    prices = np.array(prices, dtype=float)
    out = []

    for p in points:
        t = (p - base).total_seconds()

        # 前区間
        i1 = (seconds >= t - term_sec) & (seconds < t)
        if i1.sum() < 2:
            out.append(0)
            continue
        y1 = (prices[i1] - prices[i1][0]) / prices[i1][0] * 60 * 100
        s1, _ = linear_regression(seconds[i1], y1)

        # 後区間
        i2 = (seconds > t) & (seconds <= t + term_sec)
        if i2.sum() < 2:
            out.append(0)
            continue
        y2 = (prices[i2] - prices[i2][0]) / prices[i2][0] * 60 * 100
        s2, _ = linear_regression(seconds[i2], y2)

        out.append(1 if s1 >= 0 and s2 > 0 else 0)

    return out    
 
 
def read_data(symbol, year, month, week):
    filepath = f"../MarketData/Axiory/{symbol}/Tick/{symbol}_TICK_{year}-{str(month).zfill(2)}-{week}.csv" 
    df = pd.read_csv(filepath)
    df['jst'] = pd.to_datetime(df['jst'], format='ISO8601')
    return df    
    
    
    
def test(symbol, th):   
    month = 6
    for week in [1, 2, 3, 4]:
        plot(symbol, 2025, month, week, th)
    
def plot(symbol, year, month, week, th):
    df = read_data(symbol, year, month, week)
    jst = df['jst'].to_numpy()
    bid = df['bid'].to_numpy()
    
    if len(df) == 0:
        print('no data', symbol)
        return
    
    drops, spikes = detect_drop_spike(df, jst, bid, jerk_spike_th=th, jerk_drop_th=th)
    drop, drop_time, drop_value = drops
    bottom = detect_bottom(jst, bid, drop_time)
    spike, spike_time, spike_value = spikes
    rebound(df, jst, bid)
    df1 = df[df['rebound_ratio'] < 0.15]
    
    chart = TimeChart('▼: drop  ▲: spike  x: bottom, 〇: Rebound', 1800, 600, jst)
    chart.line(bid, color='blue', alpha=0.6)
    
    print(len(jst), len(bid))
    

    for t, p, b in zip(drop_time, drop_value, bottom):
        if b == 0:
            marker = 'v'
            color = 'red'
        else:
            marker = 'x'
            color='orange'
        chart.marker(t, p, marker=marker, color=color, alpha=0.5, size=20)
    
    for t, v in zip(spike_time, spike_value):
        chart.marker(t, v, marker='^', color='green', alpha=0.5, size=20)
    
    for i in range(len(df1)):
        chart.marker(df1['jst'].iloc[i], df1['bid'].iloc[i], marker='^', color='green', alpha=0.4, size=20)
    
    chart.to_png(f'./debug/{symbol}_drop_spike_{year}-{str(month).zfill(2)}-{week}.png')
    
    
def test2():
    df = read_data('CL')
    df = df.iloc[:500]
    df.to_csv('./debug/cl_tick_data.csv', index=False)
    
    
    
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    #test('NSDQ')
    #test('XAUUSD')
    test('NIKKEI', 0.05)
    #test('DOW')    
    

