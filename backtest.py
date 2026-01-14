# -----------------------------------------------------------------------------------------
# [檔案功能說明]
# 這個檔案是「回測引擎 (Backtesting Engine)」。
# 它的功能是模擬真實的市場環境，逐行讀取 K 線資料，執行策略信號，並計算損益。
# 它是完全獨立的，不連接交易所，只在本地運算。
# -----------------------------------------------------------------------------------------

# [Import 說明]
from util import *
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

# 設定日誌輸出的等級為 INFO。
logging.basicConfig(level=logging.INFO)

# [Class 說明]
# 職責：回測的主控制器 (Controller)。
class backtest:
    """只負責呼叫其他方法"""
    def __init__(self, df, config):
        self.df = df
        self.config = config
        # 將 get 到的值直接轉為 float
        self.max_hold = config["基本設定"].get("max_hold", None)
        if self.max_hold is not None:
            self.max_hold = int(self.max_hold)

        self.position = position(self.config)
        self.stats = stats(self.config)

    def _create_order(self, close: float, direction: int, timestamp, i: int):
        """內部方法，根據方向創建訂單"""
        if direction not in [1, -1]:
            raise ValueError("direction 必須是 1 (多) 或 -1 (空)")

        # --- 計算止盈 (TP) ---
        tp_value = float(self.config["止盈止損設定"]["tp_value"])
        if self.config["止盈止損設定"]["tp_of_percent"]:
            tp = close * (1 + direction * tp_value / 100)
        else:
            tp = close + direction * tp_value

        # --- 計算止損 (SL) ---
        sl_value = float(self.config["止盈止損設定"]["sl_value"])
        if self.config["止盈止損設定"]["sl_of_percent"]:
            sl = close * (1 - direction * sl_value / 100)
        else:
            sl = close - direction * sl_value
        
        # --- 計算下單數量 (Size) ---
        order_val = float(self.config["下單設定"]["order_value"])
        if self.config["下單設定"]["order_mode"] == "percent":
            base_size = self.stats.cash * order_val / 100 / close
        elif self.config["下單設定"]["order_mode"] == "price":
            base_size = order_val / close
        elif self.config["下單設定"]["order_mode"] == "fixed":
            base_size = order_val
        else:
            raise ValueError("order_mode只能是 percent, price, fixed 其中一種")

        leverage = float(self.config["下單設定"]["leverage"])
        size = float(direction) * base_size * leverage

        # --- 資金不足判斷 ---
        margin = close * base_size
        estimated_fee = margin * leverage * self.position.fee_rate
        
        if self.stats.cash < (margin + estimated_fee):
            logging.warning(f"[INSUFFICIENT_FUNDS] 時間: {timestamp}, 資金 {self.stats.cash:.2f} 不足，無法開倉 (需要 {margin+estimated_fee:.2f})")
            return 

        logs_to_process = self.position.open(close, size, tp, sl, timestamp, i)

        if logs_to_process:
            for pnl, log_event in logs_to_process:
                self.stats.trade_log(pnl, log_event)

    def show(self):
        logging.info(f"===== 績效總結 =====")
        logging.info(f"總交易次數: {self.stats.count}, 總損益: {self.stats.pnl:.2f}, 最終資金: {self.stats.cash:.2f}, 最大回撤: {self.stats.max_drawdown:.2f}")
        logging.info(f"勝率 (總/多/空): {self.stats.winrate():.2f}% / {self.stats.long_winrate():.2f}% / {self.stats.short_winrate():.2f}%")
        logging.info(f"獲利因子: {self.stats.profit_factor():.2f}, 夏普比率: {self.stats.sharpe():.2f}, 索提諾比率: {self.stats.sortino_ratio():.2f}, 卡瑪比率: {self.stats.calmar_ratio():.2f}")

    def plot_results(self):
        """繪製並儲存資金曲線圖"""
        self.stats.plot_equity_curve()

    def run(self, progress_callback=None):
        if self.df.empty:
            logging.info("DataFrame is empty, skipping backtest.")
            return

        # --- 將 DataFrame 欄位轉為 NumPy 陣列以加速存取 ---
        closes = self.df['close'].to_numpy(dtype=float)
        signals = self.df['signal'].to_numpy(dtype=np.int8)
        close_times = self.df['close_time'].to_numpy()
        
        total_len = len(self.df)
        
        # for i in tqdm(range(total_len)): # CLI環境下的進度條
        for i in range(total_len):
            if progress_callback and (i % (total_len // 100 + 1) == 0):
                progress_callback(i / total_len)

            if self.stats.cash <= 0:
                logging.info("資金不足，無法繼續交易")
                break
            
            current_close = closes[i]
            current_time = close_times[i]
            
            if self.position.size != 0:
                # 1. 檢查止損
                logs_sl = self.position.trigger_SL(current_close, current_time)
                if logs_sl:
                    for pnl, log_event in logs_sl:
                        self.stats.trade_log(pnl, log_event)

                # 2. 檢查持倉時間上限
                if self.max_hold is not None and (i - self.position.entry_index) >= self.max_hold:
                    logs_force_close = self.position.close_all(current_close, current_time)
                    if logs_force_close:
                        for pnl, log_event in logs_force_close:
                            self.stats.trade_log(pnl, log_event)

                # 3. 檢查止盈
                logs_tp = self.position.trigger_TP(current_close, current_time)
                if logs_tp:
                    for pnl, log_event in logs_tp:
                        self.stats.trade_log(pnl, log_event)
            
            # 4. 檢查進場信號
            current_signal = signals[i]
            if current_signal == 1:
                self._create_order(current_close, 1, current_time, i)
            elif current_signal == -1:
                self._create_order(current_close, -1, current_time, i)
        
        # --- 迴圈結束後處理 ---
        if self.position.size != 0:
            last_price = closes[-1]
            last_time = close_times[-1]
            close_results = self.position.close_all(last_price, last_time)
            if close_results:
                pnl, log = close_results[0]
                self.stats.trade_log(pnl, log)
        
        # --- 一次性生成 Log DataFrame ---
        self.stats.finalize_log()

        if progress_callback:
            progress_callback(1.0)
        
        logging.info(f"總交易次數: {self.stats.count}, 總損益: {self.stats.pnl:.2f}, 最大回撤: {self.stats.max_drawdown:.2f}, 最終資金: {self.stats.cash:.2f}")
        
        if not self.stats.log.empty:
            result_to_csv(self.stats.log, is_backtest=True)

# [Class 說明]
# 職責：管理單一倉位 (Position) 的狀態。
class position:
    """只負責倉位部分的動作"""
    def __init__(self, config):
        self.config = config
        self.avg_price = 0.0
        self.size = 0.0
        self.tp = 0.0
        self.sl = 0.0
        self.entry_index = 0
        self.fee_rate = float(config["回測設定"]["fee_rate"])
        self.slippage = float(config["回測設定"]["slippage"])
        self.allow_pyramiding = bool(config["下單設定"]["pyramiding"])
        self.allow_reverse = bool(config["下單設定"]["reverse"])


    def open(self, price: float, size: float, tp: float, sl: float, timestamp, entry_index: int) -> list[tuple[float, dict]]:
        """開倉，回傳 (pnl, log_dict) 的列表"""
        if abs(size) < 1e-12: # 檢查 size 是否趨近於 0
            raise ValueError("size不能為0")

        direction = 1 if size > 0 else -1
        
        # 根據滑價調整實際成交價
        if direction == 1: # 做多
            if not (sl is None or tp is None or sl < price < tp):
                 if sl is not None and tp is not None and not (sl < price < tp):
                    raise ValueError(f"多單必須符合 止損<價格<止盈 {sl}<{price}<{tp}")
            price *= (1 + self.slippage)
        else: # 做空
            if not (sl is None or tp is None or tp < price < sl):
                if sl is not None and tp is not None and not (tp < price < sl):
                    raise ValueError(f"空單必須符合 止盈<價格<止損 {tp}<{price}<{sl}")
            price *= (1 - self.slippage)
        
        if self.size != 0:
            # 反向開單
            if self.size * size < 0:
                if self.allow_reverse:
                    return self.reverse(price, size, tp, sl, timestamp, entry_index)
                return [] 
            # 同向開單 (加倉)
            else:
                if not self.allow_pyramiding:
                    return [] 
        
        # 計算新均價和總倉位
        self.avg_price = (self.avg_price * abs(self.size) + price * abs(size)) / (abs(self.size) + abs(size))
        self.size += size
        self.tp = tp
        self.sl = sl
        self.entry_index = entry_index 

        log = {
            "時間": timestamp, "狀態": "開倉", "多/空": direction, 
            "進場價": price, "進場量": size, "當前均價": self.avg_price, "當前持倉量": self.size
        }
        return [(0.0, log)] 

    def close(self, price: float, size_to_close: float, timestamp) -> list[tuple[float, dict]]:
        """平倉，回傳 (pnl, log_dict) 的列表"""
        if self.size == 0:
            raise Exception("當前無持倉，無法平倉")

        direction_to_close = 1 if size_to_close > 0 else -1
        
        # 根據滑價調整實際成交價
        if direction_to_close == 1: # 買入平倉(空單)
            price *= (1 + self.slippage) 
        else: # 賣出平倉(多單)
            price *= (1 - self.slippage) 

        
        # --- 根據多空方向，使用不同的損益計算邏輯 ---
        if self.size > 0: # 原多單，賣出平倉
            gross_pnl = (price - self.avg_price) * abs(size_to_close)
        else: # 原空單，買入平倉
            gross_pnl = (self.avg_price - price) * abs(size_to_close)
            
        closing_fee = abs(size_to_close) * price * self.fee_rate
        pnl = gross_pnl - closing_fee
        
        self.size += size_to_close
        if abs(self.size) < 1e-9: # 避免浮點數問題
            self.size = 0.0
            self.avg_price = 0.0
            
        log = {
            "時間": timestamp, "狀態": "平倉", "出場價": price, 
            "出場量": size_to_close, "實現損益": pnl, "剩餘倉位": self.size
        }
        return [(pnl, log)]
    
    def close_all(self, price: float, timestamp) -> list[tuple[float, dict]]:
        """全部平倉"""
        if self.size == 0:
            return []
        size_to_close = -self.size
        self.sl = None
        self.tp = None
        return self.close(price, size_to_close, timestamp)

    def reverse(self, price: float, new_size: float, tp: float, sl: float, timestamp, entry_index:int) -> list[tuple[float, dict]]:
        """反手"""
        close_results = self.close_all(price, timestamp)
        if not close_results:
            return []
        
        pnl_close, log_close = close_results[0]
        
        # 開新倉
        direction = 1 if new_size > 0 else -1
        self.avg_price = price
        self.size = new_size
        self.tp = tp
        self.sl = sl
        self.entry_index = entry_index
        
        log_open = {
            "時間": timestamp, "狀態": "開倉", "多/空": direction, "進場價": price, 
            "進場量": new_size, "當前均價": self.avg_price, "當前持倉量": self.size
        }
        
        return [(pnl_close, log_close), (0.0, log_open)]

    def trigger_SL(self, close: float, timestamp):
        if self.sl is not None and self.sl != 0:
            # 使用 isclose 來處理浮點數精度問題
            is_close = math.isclose(close, self.sl)
            if (self.size > 0 and (close <= self.sl or is_close)) or \
               (self.size < 0 and (close >= self.sl or is_close)):
                return self.close_all(self.sl, timestamp)
        return []
    
    def trigger_TP(self, close: float, timestamp):
        if self.tp is not None and self.tp != 0:
            # 使用 isclose 來處理浮點數精度問題
            is_close = math.isclose(close, self.tp)
            if (self.size > 0 and (close >= self.tp or is_close)) or \
               (self.size < 0 and (close <= self.tp or is_close)):
                return self.close_all(self.tp, timestamp)
        return []

# [Class 說明]
# 職責：統計與會計模組 (Statistics)。
class stats:
    """只負責記錄資料的動作"""
    def __init__(self, config):
        self.config = config
        self.count = 0
        self.count_long = 0
        self.count_long_win = 0
        self.count_short = 0
        self.count_short_win = 0
        
        self.log_events = [] # 改為儲存字典列表
        self.log = pd.DataFrame()
        
        self.cash = float(config["回測設定"]["initial_cash"])
        self.pnl = 0.0
        self.peak_equity = self.cash
        self.max_drawdown = 0.0

    def trade_log(self, pnl: float, log_event: dict):
        if log_event is not None:
            self.log_events.append(log_event)

            if log_event.get("狀態") == "開倉":
                fee_rate = float(self.config["回測設定"]["fee_rate"])
                opening_fee = abs(log_event["進場量"]) * log_event["進場價"] * fee_rate
                self.cash -= opening_fee
                self.pnl -= opening_fee

            elif log_event.get("狀態") == "平倉":
                if pnl is None:
                    return

                self.cash += pnl
                # 更新 PnL & Drawdown
                current_equity = self.cash
                self.pnl += pnl # 這裡的pnl是已經扣掉手續費的
                self.peak_equity = max(self.peak_equity, current_equity)
                drawdown = self.peak_equity - current_equity
                self.max_drawdown = max(self.max_drawdown, drawdown)

                closed_size = log_event["出場量"]
                self.count += 1
                
                # 根據出場量正負判斷原單方向
                if closed_size > 0: # 買入平倉 (原空單)
                    self.count_short += 1
                    if pnl > 0: self.count_short_win += 1
                else: # 賣出平倉 (原多單)
                    self.count_long += 1
                    if pnl > 0: self.count_long_win += 1

    def finalize_log(self):
        """在回測結束後，一次性生成最終的 DataFrame。"""
        if self.log_events:
            self.log = pd.DataFrame(self.log_events)
    
    def sharpe(self):
        if self.log.empty or "實現損益" not in self.log.columns or self.log["實現損益"].isnull().all():
            return 0.0 

        temp_log = self.log.copy()
        temp_log['時間'] = pd.to_datetime(temp_log['時間'])
        temp_log.set_index('時間', inplace=True)

        daily_returns = temp_log['實現損益'].resample('D').sum()

        if len(daily_returns) < 2 or daily_returns.std() == 0:
            return 0.0

        rf_per_day = 0.01 / 252
        avg_return = daily_returns.mean()
        std_return = daily_returns.std()
        
        sharpe_ratio = (avg_return - rf_per_day) / std_return * np.sqrt(252)
        return sharpe_ratio

    def get_equity_curve(self):
        """
        計算資金曲線與水下圖數據。
        返回一個包含 '時間', '資金曲線', '回撤' 的 DataFrame。
        """
        if self.log.empty or "實現損益" not in self.log.columns or self.log["實現損益"].dropna().empty:
            return None

        pnl_events = self.log[self.log['實現損益'].notna()].copy()
        pnl_events['時間'] = pd.to_datetime(pnl_events['時間'])
        pnl_events = pnl_events.sort_values(by='時間')
        
        initial_cash = float(self.config["回測設定"]["initial_cash"])
        pnl_events['累計損益'] = pnl_events['實現損益'].cumsum()
        pnl_events['資金曲線'] = initial_cash + pnl_events['累計損益']
        
        # 計算水下圖 (Drawdown)
        pnl_events['滾動高點'] = pnl_events['資金曲線'].expanding().max()
        # 計算回撤百分比
        pnl_events['回撤'] = (pnl_events['資金曲線'] - pnl_events['滾動高點']) / pnl_events['滾動高點']
        
        return pnl_events[['時間', '資金曲線', '回撤']]

    def get_periodic_returns(self, period='M'):
        """
        計算週期性報酬。
        :param period: 'M' for Monthly, 'A' for Annual, 'Q' for Quarterly
        :return: A DataFrame with periodic returns.
        """
        if self.log.empty or "實現損益" not in self.log.columns or self.log["實現損益"].dropna().empty:
            return None
        
        pnl_log = self.log[self.log['實現損益'].notna()].copy()
        pnl_log['時間'] = pd.to_datetime(pnl_log['時間'])
        pnl_log = pnl_log.set_index('時間')
        
        periodic_returns = pnl_log['實現損益'].resample(period).sum()
        
        # 轉換為 DataFrame 並重設索引，方便 Plotly 處理
        periodic_returns = periodic_returns.reset_index()
        periodic_returns.columns = ['時間', '損益']
        
        return periodic_returns

    def plot_equity_curve(self):
        """繪製資金曲線圖"""
        equity_df = self.get_equity_curve()
        if equity_df is None:
            logging.warning("沒有足夠的交易數據來繪製資金曲線圖。")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(equity_df['時間'], equity_df['資金曲線'], label='Equity Curve')
        plt.title(f"Equity Curve - {self.config['基本設定']['symbol']}")
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.legend()
        plt.show()

    def long_winrate(self):
        if self.count_long == 0: return 0.0
        return (self.count_long_win / self.count_long) * 100

    def short_winrate(self):
        if self.count_short == 0: return 0.0
        return (self.count_short_win / self.count_short) * 100

    def winrate(self):
        if self.count == 0: return 0.0
        return ((self.count_long_win + self.count_short_win) / self.count) * 100

    def profit_factor(self):
        """計算獲利因子"""
        if self.log.empty or "實現損益" not in self.log.columns:
            return 0.0

        returns = self.log["實現損益"].dropna()
        total_profit = returns[returns > 0].sum()
        total_loss = abs(returns[returns < 0].sum())

        if total_loss == 0:
            return np.inf if total_profit > 0 else 0.0
        return total_profit / total_loss

    def sortino_ratio(self):
        """計算年化索提諾比率"""
        if self.log.empty or "實現損益" not in self.log.columns or self.log["實現損益"].isnull().all():
            return 0.0

        temp_log = self.log.copy()
        temp_log['時間'] = pd.to_datetime(temp_log['時間'])
        temp_log.set_index('時間', inplace=True)
        daily_returns = temp_log['實現損益'].resample('D').sum()

        if len(daily_returns) < 2: return 0.0

        rf_per_day = 0.01 / 252
        mean_daily_return = daily_returns.mean()
        
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = negative_returns.std()

        if downside_std == 0 or pd.isna(downside_std):
            return 0.0

        sortino = (mean_daily_return - rf_per_day) / downside_std * np.sqrt(252)
        return sortino

    def calmar_ratio(self):
        """計算卡瑪比率"""
        if self.log.empty or "實現損益" not in self.log.columns or self.log["實現損益"].isnull().all():
            return 0.0
        
        initial_cash = float(self.config["回測設定"]["initial_cash"])
        if initial_cash == 0: return 0.0
        
        # 計算年化報酬率
        temp_log = self.log.copy()
        temp_log['時間'] = pd.to_datetime(temp_log['時間'])
        start_date = temp_log['時間'].min()
        end_date = temp_log['時間'].max()
        num_days = (end_date - start_date).days
        if num_days < 1: return 0.0

        total_return_rate = self.pnl / initial_cash
        annualized_return = (1 + total_return_rate) ** (365.0 / num_days) - 1

        # 最大回撤值 (這裡是負數，取絕對值)
        max_dd_value = abs(self.max_drawdown)
        if max_dd_value == 0:
            return np.inf if annualized_return > 0 else 0.0

        return annualized_return / (max_dd_value / initial_cash)

if __name__ == "__main__":
    config = load_config()
    # 確保 if __name__ == "__main__": 下的路徑和檔案名稱生成邏輯正確
    import re
    # 檔案名稱中的時間格式可能包含不適合做檔名的 ":" 符號，需要替換
    start_time_str = re.sub(":", "-", config['回測設定']['start_time'])
    end_time_str = re.sub(":", "-", config['回測設定']['end_time'])
    # 從 util.py 取得 get_processed_data 函式 (假設它存在)
    df = get_processed_data(f"{config['基本設定']['symbol']}_{config['基本設定']['timeframe']}_{config['基本設定']['strategy']}_{start_time_str} to {end_time_str}.csv")

    if df is not None:
        bt = backtest(df, config)
        bt.run()
        bt.show()
        bt.plot_results()
    else:
        print("找不到對應的 Processed Data，請先執行一次回測來產生。")