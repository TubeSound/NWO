
import MetaTrader5 as mt5api
import pandas as pd
from dateutil import tz
from datetime import datetime, timedelta
import numpy as np
from dateutil import tz
from common import Signal, TimeFrame, Columns

JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')  


order_types = {
                mt5api.ORDER_TYPE_BUY:'Market Buy order',
                mt5api.ORDER_TYPE_SELL: 'Market Sell order',
                mt5api.ORDER_TYPE_BUY_LIMIT: 'Buy Limit pending order',
                mt5api.ORDER_TYPE_SELL_LIMIT: 'Sell Limit pending order',
                mt5api.ORDER_TYPE_BUY_STOP: 'Buy Stop pending order',
                mt5api.ORDER_TYPE_SELL_STOP:'Sell Stop pending order',
                mt5api.ORDER_TYPE_BUY_STOP_LIMIT: 'Upon reaching the order price, a pending Buy Limit order is placed at the StopLimit price',
                mt5api.ORDER_TYPE_SELL_STOP_LIMIT: 'Upon reaching the order price, a pending Sell Limit order is placed at the StopLimit price',
                mt5api.ORDER_TYPE_CLOSE_BY: 'Order to close a position by an opposite one'
}   

        
def now():
    t = datetime.now(tz=UTC)
    return t

# numpy timestamp -> pydatetime naive
def nptimestamp2pydatetime(npdatetime):
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    seconds_since_epoch = (npdatetime - unix_epoch) / one_second
    dt = datetime.utcfromtimestamp(seconds_since_epoch)
    #timestamp = (npdatetime - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    #dt2 = datetime.utcfromtimestamp(timestamp)
    return dt



def slice(df, ibegin, iend):
    new_df = df.iloc[ibegin: iend + 1, :]
    return new_df

def df2dic(df: pd.DataFrame):
    dic = {}
    for column in df.columns:
        dic[column] = df[column].to_numpy()
    return dic

def time_str_2_datetime(df, time_column, format='%Y-%m-%d %H:%M:%S'):
    time = df[time_column].to_numpy()
    new_time = [datetime.strptime(t, format) for t in time]
    df[time_column] = new_time
    
def position_dic_array(positions):
    array = []
    for position in positions:
        d = {
            'ticket': position.ticket,
            'time': pd.to_datetime(position.time, unit='s'),
            'symbol': position.symbol,
            'type': position.type,
            'volume': position.volume,
            'price_open': position.price_open,
            'sl': position.sl,
            'tp': position.tp,
            'price_current': position.price_current,
            'profit': position.profit,
            'swap': position.swap,
            'comment': position.comment,
            'magic': position.magic}
        array.append(d)
    return array

class PositionInfo:
    def __init__(self, symbol, typ, index: int, time: datetime, volume, ticket, price, sl, tp, target_profit=0):
        self.symbol = symbol
        self.type = typ
        self.volume = volume
        self.ticket = ticket
        self.entry_index = index
        self.entry_time = time
        self.entry_price = price
        self.sl = sl
        self.tp = tp
        self.target_profit = target_profit

        self.exit_index = None
        self.exit_time = None       
        self.exit_price = None
        self.profit = None
        self.profit_max = None
        self.closed = False
        self.losscutted = False
        self.trailing_stopped = False
        self.time_upped = False
 
    def signal(self):
        if self.type == mt5api.ORDER_TYPE_BUY:
            return Signal.LONG
        elif self.type == mt5api.ORDER_TYPE_BUY_LIMIT:
            return Signal.LONG
        elif self.type == mt5api.ORDER_TYPE_BUY_STOP:
            return Signal.LONG
        elif self.type == mt5api.ORDER_TYPE_BUY_STOP_LIMIT:
            return Signal.LONG
        elif self.type == mt5api.ORDER_TYPE_SELL:
            return Signal.SHORT
        elif self.type == mt5api.ORDER_TYPE_SELL_LIMIT:
            return Signal.SHORT
        elif self.type == mt5api.ORDER_TYPE_SELL_STOP:
            return Signal.SHORT
        elif self.type == mt5api.ORDER_TYPE_SELL_STOP_LIMIT:
            return Signal.SHORT
        else:
            return None
 
    def update_profit(self, price):
        profit = price - self.entry_price
        if self.signal() == Signal.SHORT:
            profit *= -1
        triggered = False
        if self.profit_max is None:
            if profit >= self.target_profit:
                self.profit_max = profit
                triggered = True
        else:
            if profit > self.profit_max:
                self.profit_max = profit
        return triggered, profit, self.profit_max
        
    def timeup_count(self, timelimit: int):
        if self.is_no_takeprofit():
            self.timelimit = timelimit
                
    def timeup(self, index: int):
        return ((index - self.entry_index) > self.timelimit)
            
    def desc(self):
        type_str = order_types[self.type]
        s = 'symbol: ' + self.symbol + ' type: ' + type_str + ' volume: ' + str(self.volume) + ' ticket: ' + str(self.ticket)
        return s
    
    def array(self):
        columns = ['symbol', 'type', 'volume', 'ticket', 'sl', 'entry_time', 'entry_price', 'exit_time', 'exit_price', 'profit', 'closed', 'losscutted', 'trailing_stopped', 'time_upped']
        type_str = order_types[self.type]
        data = [self.symbol, type_str, self.volume, self.ticket, self.sl, self.entry_time, self.entry_price, self.exit_time, self.exit_price, self.profit, self.closed, self.losscutted, self.trailing_stopped, self.time_upped]
        return data, columns
    
class Mt5Trade:
    def __init__(self):
        self.ticket = None
        self.symbol = None
        
    @staticmethod
    def connect():
        if mt5api.initialize():
            print('Connected to MT5 Version', mt5api.version())
        else:
            print('initialize() failed, error code = ', mt5api.last_error())
            
    def set_symbol(self, symbol):
        self.symbol = symbol
        
    def parse_order_result(self, result, index: int, time: datetime, stoploss, takeprofit):
        if result is None:
            print('Error')
            return False, None
        
        if takeprofit is None:
            takeprofit = 0
        code = result.retcode
        if code == 10009:
            #print("注文完了", self.symbol, 'type', result.request.type, 'volume', result.volume)
            position_info = PositionInfo(self.symbol, result.request.type, index, time, result.volume, result.order, result.price, stoploss, takeprofit)
            return True, position_info
        elif code == 10013:
            print("無効なリクエスト")
            return False, None
        elif code == 10018:
            print("マーケットが休止中")
            return False, None       
        elif code == 10019:
            print("リクエストを完了するのに資金が不充分。")
            return False, None
        else:
            print('Entry error code', code)
            return False, None
        
    def current_price(self, signal):
        tick = mt5api.symbol_info_tick(self.symbol)
        if signal == Signal.LONG:
            return tick.ask
        elif signal == Signal.SHORT:
            return tick.bid 
        else:
            return None
        
    def entry(self, signal: Signal, index: int, time: datetime, volume:float, stoploss=None, takeprofit=0, deviation=20):        
        point = mt5api.symbol_info(self.symbol).point
        price = self.current_price(signal)
        if signal == Signal.LONG:
            typ =  mt5api.ORDER_TYPE_BUY
        elif signal == Signal.SHORT:
            typ =  mt5api.ORDER_TYPE_SELL
            
        request = {
            "action": mt5api.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(volume),
            "type": typ,
            "price": float(price),
            "deviation": deviation,# 許容スリップページ
            "magic":  234000,
            "comment": "python script open",
            "type_time": mt5api.ORDER_TIME_GTC,
            "type_filling": mt5api.ORDER_FILLING_IOC,
        }

        if stoploss > 0:
            if signal == Signal.LONG:
                request['sl'] = float(price - stoploss)
            elif signal == Signal.SHORT:
                request['sl'] = float(price + stoploss)
        
        if takeprofit is None:
            takeprofit = 0
        if takeprofit > 0:
            if signal == Signal.LONG:
                request['tp'] = float(price + takeprofit)
            elif signal == Signal.SHORT:
                request['tp'] = float(price - takeprofit)
        result = mt5api.order_send(request)
        #print('エントリー ', request)
        return self.parse_order_result(result, index, time, stoploss, takeprofit)
    
    def get_positions(self):
        positions = mt5api.positions_get(symbol=self.symbol)
        if positions is None:
            raise Exception('get position error')
        return positions

    def is_long(self, typ):
        if typ == mt5api.ORDER_TYPE_BUY or typ == mt5api.ORDER_TYPE_BUY_LIMIT or typ == mt5api.ORDER_TYPE_BUY_STOP_LIMIT:
            return True
        else:
            return False

    def is_short(self, typ):
        if typ == mt5api.ORDER_TYPE_SELL or typ == mt5api.ORDER_TYPE_SELL_LIMIT or typ == mt5api.ORDER_TYPE_SELL_STOP_LIMIT:
            return True
        else:
            return False

    def close_position(self, position, volume=None, deviation=20):
        if volume is None:
            volume = position.volume        
        tick = mt5api.symbol_info_tick(position.symbol)
        if self.is_long(typ):
            price = tick.bid
            typ = mt5api.ORDER_TYPE_SELL
        elif self.is_short(typ):
            price = tick.ask
            typ = mt5api.ORDER_TYPE_BUY
        return self.close(typ, position.ticket, price, volume, deviation=deviation)
    
    def close_order_result(self, info: PositionInfo, volume=None, deviation=20):
        if volume is None:
            volume = info.volume        
        tick = mt5api.symbol_info_tick(info.symbol)
        if self.is_long(info.type):
            price = tick.bid
            typ = mt5api.ORDER_TYPE_SELL
        elif self.is_short(info.type):
            price = tick.ask
            typ = mt5api.ORDER_TYPE_BUY
        return self.close(typ, info.ticket, price, volume, deviation=deviation)

    def close_by_position_info(self, position_info: PositionInfo):
        tick = mt5api.symbol_info_tick(position_info.symbol)            
        if self.is_long(position_info.type):
            price = tick.bid
            typ = mt5api.ORDER_TYPE_SELL
        else:
            price = tick.ask
            typ = mt5api.ORDER_TYPE_BUY
        return self.close(typ, position_info.ticket, price, position_info.volume)

    def close(self, typ, ticket, price, volume, deviation=20):
        request = {
            "action": mt5api.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": self.symbol,
            "volume": volume,
            "type": typ,
            "price": price,
            "deviation": deviation,
            "magic": 100,
            "comment": "python script close",
            "type_time": mt5api.ORDER_TIME_GTC,
            "type_filling": mt5api.ORDER_FILLING_IOC,
        }
        result = mt5api.order_send(request)
        #print('決済', request)
        return self.parse_order_result(result, None, None, None, None)
    
    def close_all_position(self):
        positions = self.get_positions()
        for position in positions:
            self.close_position(position, position.volume)
    
    def get_ticks_jst(self, jst_begin, jst_end):
        t_begin = self.jst2utc(jst_begin)
        t_end = self.jst2utc(jst_end)
        return self.get_ticks(t_begin, t_end)

    def get_ticks(self, symbol, utc_begin, utc_end):
        ticks = mt5api.copy_ticks_range(symbol, utc_begin, utc_end, mt5api.COPY_TICKS_ALL)
        return self.parse_ticks(ticks)    
    
    def get_ticks_from(self, symbol,  utc_time, length=10):
        ticks = mt5api.copy_ticks_from(symbol, utc_time, length, mt5api.COPY_TICKS_ALL)
        return self.parse_ticks(ticks)
        
    def parse_ticks(self, ticks):
        df = pd.DataFrame(ticks)
        df[Columns.TIME] = pd.to_datetime(df[Columns.TIME], unit='s')
        return df
    
    def get_rates_jst(self, symbol, timeframe: TimeFrame, jst_begin, jst_end):
        t_begin = self.jst2utc(jst_begin)
        t_end = self.jst2utc(jst_end)
        return self.get_rates_utc(symbol, timeframe, t_begin, t_end)
    
    def get_rates_utc(self, symbol, timeframe, utc_begin, utc_end):
        rates = mt5api.copy_rates_range(symbol, TimeFrame.const(timeframe), utc_begin, utc_end)
        return self.parse_rates(rates)
        
    def get_rates(self, symbol, timeframe: str, length: int):
        #print(self.symbol, timeframe)
        rates = mt5api.copy_rates_from_pos(symbol, TimeFrame.const(timeframe), 0, length)
        if rates is None:
            raise Exception('get_rates error')
        return self.parse_rates(rates)

    def parse_rates(self, rates):
        df = pd.DataFrame(rates)
        df[Columns.TIME] = pd.to_datetime(df[Columns.TIME], unit='s')
        return df
        
class Mt5TradeSim:
    def __init__(self, symbol: str, files: dict):
        self.symbol = symbol
        self.load_data(files)
                
    def adjust_msec(self, df, time_column, msec_column):
        new_time = []
        for t, tmsec in zip(df[time_column], df[msec_column]):
            msec = tmsec % 1000
            dt = t + timedelta(milliseconds= msec)
            new_time.append(dt)
        df[time_column] = new_time
                
    def load_data(self, files):
        dic = {}
        for timeframe, file in files.items():
            df = pd.read_csv(file)
            time_str_2_datetime(df, Columns.TIME)
            if timeframe == TimeFrame.TICK:
                self.adjust_msec(df, Columns.TIME, 'time_msc')
            dic[timeframe] = df
        self.dic = dic
    
    def search_in_time(self, df, time_column, utc_time_begin, utc_time_end):
        time = list(df[time_column].values)
        if utc_time_begin is None:
            ibegin = 0
        else:
            ibegin = None
        if utc_time_end is None:
            iend = len(time) - 1
        else:
            iend = None
        for i, t in enumerate(time):
            dt = npdatetime2datetime(t)
            if ibegin is None:
                if dt >= utc_time_begin:
                    ibegin = i
            if iend is None:
                if dt > utc_time_end:
                    iend = i - 1
            if ibegin is not None and iend is not None:
                break
        slilced = slice(df, ibegin, iend)
        return slilced

    def get_rates(self, timeframe: str, utc_begin, utc_end):
        #print(self.symbol, timeframe)
        df = self.dic[timeframe]
        return self.search_in_time(df, Columns.TIME, utc_begin, utc_end)

    def get_ticks(self, utc_begin, utc_end):
        df = self.dic[TimeFrame.TICK]
        return self.search_in_time(df, Columns.TIME, utc_begin, utc_end)

def test1():
    symbol = 'NIKKEI'
    mt5trade1 = Mt5Trade(symbol)
    mt5trade2 = Mt5Trade('DOW')
    Mt5Trade.connect()
    t = datetime.now().astimezone(JST)
    ret, result = mt5trade1.entry(Signal.SHORT, 0, t, 0.1, stoploss=300.0)
    if ret:
        result.desc()
    ret, result = mt5trade2.entry(Signal.LONG, 0, t, 0.1, stoploss=300.0)
    if ret:
        result.desc()
    
    
    mt5trade.close_order_result(result, result.volume)
    pass


if __name__ == '__main__':
    test1()
