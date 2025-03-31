import os
import shutil
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.plotting import show
from bokeh.plotting import output_notebook
from bokeh.layouts import column, row, layout, gridplot
from bokeh.models import Spacer, Span, Text
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.io import export_png
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

output_notebook()

TOOLS = "pan,wheel_zoom,box_zoom,box_select,crosshair,reset,save"
TOOLTIPS=[  ( 'date',   '@date' ),
            ( 'close',  '@close{0,0}' ), 
        ]

class TimeChart():
    def __init__(self, title, width, height, time, date_format='%Y/%m/%d %H:%M'):
        self.fig = figure(  x_axis_type="linear",
                            tools=TOOLS, 
                            plot_width=width,
                            plot_height=height,
                            tooltips=TOOLTIPS,
                            title = title)
    
        self.time = list(time)
        dt = time[-1] - time[-2]
        for i in range(1, 50):
            time.append(time[-1] + i * dt)
        
        self.fig.xaxis.major_label_overrides = {i: d.strftime(date_format) for i, d in enumerate(time)}
        self.indices = range(len(time))

    def time_index(self, time):
        for i, d in enumerate(self.time):
            if d > time:
                return i
        return -1      

    def line(self, y, **kwargs):
        self.fig.line(self.indices, np.array(y), **kwargs)
        
    def scatter(self, ts, ys, **kwargs):
        indices = [self.time_index(t) for t in ts]
        self.fig.scatter(indices, np.array(ys), **kwargs)
    
        
    def markers(self, signal, values, status, marker='o', color='black', size=10):
        marks = {'o': 'circle', 'v': 'inversed_triangle', '^': 'triangle', '+': 'cross', 'x': 'x'}
        indices = []
        ys = []
        for i, (s, v) in enumerate(zip(signal, values)):
            if s == status:
                indices.append(i)
                ys.append(v)
        self.fig.scatter(indices, np.array(ys), marker=marks[marker], color=color, size=size)
        
    def vline(self, time, color, width=1):
        index = self.time_index(time)
        span = Span(location=index,
                    dimension='height',
                    line_color=color,
                    line_width=width)
        self.fig.add_layout(span)
        
    def text(self, time, y, text, color):
        glyph = Text(x="x", y="y", text="text",  text_color=color, text_font_size='9pt')
        source = ColumnDataSource(dict(x=[self.time_index(time)], y=[y], text=[text]))
        self.fig.add_glyph(source, glyph)
        
        
class CandleChart(TimeChart):
    def __init__(self, title, width, height, time, op, hi, lo, cl, date_format='%Y/%m/%d %H:%M'):
        super().__init__(title, width, height, time, date_format)
        self.op = np.array(op)
        self.hi = np.array(hi)
        self.lo = np.array(lo)
        self.cl = np.array(cl)
        self.plot_candle()
        
    def pickup(self, valid, arrays):
        out = []
        index = []
        for i, array in enumerate(arrays):
            d = []
            for j, (a, v) in enumerate(zip(array, valid)):
                if v:
                    if i == 0:
                        index.append(j)
                    d.append(a)
            out.append(np.array(d))
        return index, out
                    
    def plot_candle(self):
        ascend = self.cl > self.op
        descend = ~ascend
        self.fig.segment(self.indices, self.hi, self.indices, self.lo, color="black")
        time, values = self.pickup(ascend, [self.op, self.cl])        
        self.fig.vbar(time, 0.5, values[0], values[1], fill_color="#44a144", line_color="gray")
        time, values = self.pickup(descend, [self.op, self.cl])        
        self.fig.vbar(time, 0.5, values[0], values[1], fill_color="#F23f3E", line_color="gray")
       

def from_pickle(file):
    import pickle
    filepath = f'../BlackSwan/data/Axiory/{file}'
    with open(filepath, 'rb') as f:
        data0 = pickle.load(f)
        return data0
    return None        
        
def test():
    from common import Columns
    from time_utils import TimeUtils
    symbol = 'NIKKEI_M30'
    data0 = from_pickle(f'{symbol}.pkl')
    jst0 = data0[Columns.JST]
    t1 = jst0[-1]
    t0 = t1 - timedelta(days=2)
    n, data = TimeUtils.slice(data0, Columns.JST, t0, t1)
    jst = np.array(data[Columns.JST])
    op = np.array(data[Columns.OPEN])
    hi = np.array(data[Columns.HIGH])
    lo = np.array(data[Columns.LOW])
    cl = np.array(data[Columns.CLOSE])
    chart = CandleChart(symbol, 1200, 500, jst, op, hi, lo, cl)
    export_png(chart.fig, filename='debug.png')
    
    

if __name__ == '__main__':
    test()
