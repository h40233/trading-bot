# -----------------------------------------------------------------------------------------
# [檔案功能說明]
# 這個檔案是「回測引擎 (Backtesting Engine)」。
# 它的功能是模擬真實的市場環境，逐行讀取 K 線資料，執行策略信號，並計算損益。
# 它是完全獨立的，不連接交易所，只在本地運算。
# 核心組件包含：
# 1. backtest (類別): 總指揮，負責跑迴圈、調度資金、下單。
# 2. position (類別): 倉位管家，負責計算持倉均價、開平倉邏輯、計算手續費。
# 3. stats (類別): 會計師，負責記錄每一筆交易、繪製資金曲線、計算夏普比率等績效指標。
# -----------------------------------------------------------------------------------------

# [Import 說明]
# 從 util.py 引入工具函式 (如 load_config, get_processed_data)。
from util import *

# 引入 logging，用於顯示回測過程中的資訊 (如買賣操作、資金不足警告)。
import logging

# 引入 matplotlib.pyplot，用於繪製圖表 (資金曲線圖)。
import matplotlib.pyplot as plt

# 引入 Decimal 和 getcontext。
# Decimal: Python 的高精度數字型態，專門用於金融計算，避免浮點數誤差。
# getcontext: 用於設定 Decimal 的全域精度 (如小數點後幾位)。
from decimal import Decimal, getcontext

# 引入 numpy，用於科學計算 (主要用於計算標準差 std 和開根號 sqrt)。
import numpy as np

# 引入 tqdm，這是一個進度條套件，讓回測跑迴圈時能看到進度 (0% -> 100%)。
from tqdm import tqdm

# 設定日誌輸出的等級為 INFO。
logging.basicConfig(level=logging.INFO)

# [Class 說明]
# 職責：回測的主控制器 (Controller)。
# 它持有 K 線資料 (df) 和設定 (config)，負責推進時間 (run 迴圈)，
# 並協調 position (倉位) 和 stats (統計) 來完成回測。
class backtest:
    # 設定 Decimal 的運算精度為 28 位。
    # 這確保了在進行乘除運算時，不會因為位數不夠而產生誤差。
    getcontext().prec = 28

    """只負責呼叫其他方法"""
    # [Function 說明]
    # 功能：初始化回測物件。
    # 原理：將外部傳入的資料和設定存起來，並建立倉位和統計物件。
    def __init__(self, df, config):
        # 儲存 K 線資料 (DataFrame)。
        self.df = df
        # 儲存設定檔 (Dictionary)。
        self.config = config
        # 初始化 position 物件，負責管理當前的持倉狀態。
        self.position = position(self.config)
        # 初始化 stats 物件，負責記錄歷史交易和計算績效。
        self.stats = stats(self.config)
        # 從設定檔讀取 "max_hold" (最大持倉 K 棒數)。
        # .get() 方法：如果設定檔沒寫這欄，就預設為 None (代表不啟用此功能)。
        self.max_hold = self.config["基本設定"].get("max_hold", None) # 從設定檔讀取最長持倉K棒數

    # [Function 說明]
    # 功能：計算下單參數並執行開倉。
    # 原理：
    # 1. 計算止盈 (TP) 和止損 (SL) 的價格。
    # 2. 計算下單數量 (Size) (依據設定是固定金額還是百分比)。
    # 3. 檢查資金是否足夠 (保證金 + 手續費)。
    # 4. 呼叫 self.position.open 執行開倉。
    def _create_order(self, close: Decimal, direction: int, timestamp, i: int):
        """內部方法，根據方向創建訂單"""
        # 檢查方向參數是否合法 (只能是 1 或 -1)。
        if direction not in [1, -1]:
            raise ValueError("direction 必須是 1 (多) 或 -1 (空)")

        # --- 計算止盈 (Take Profit) ---
        # 如果設定為百分比模式 (例如賺 10% 止盈)。
        if self.config["止盈止損設定"]["tp_of_percent"]:
            # 公式：現價 * (1 + 方向 * 百分比)。
            # 注意：所有數字都先轉字串再轉 Decimal，這是避免浮點數誤差的標準寫法。
            tp = close * (Decimal('1') + Decimal(str(direction)) * Decimal(str(self.config["止盈止損設定"]["tp_value"])) / Decimal('100'))
        else:
            # 如果是固定價格模式 (例如價格加 100 點止盈)。
            # 公式：現價 + (方向 * 數值)。
            tp = close + Decimal(str(direction)) * Decimal(str(self.config["止盈止損設定"]["tp_value"]))

        # --- 計算止損 (Stop Loss) ---
        # 如果設定為百分比模式 (例如虧 5% 止損)。
        if self.config["止盈止損設定"]["sl_of_percent"]:
            # 公式：現價 * (1 - 方向 * 百分比)。
            # 這裡用減號是因為：多單(1)止損要在下方，空單(-1)止損要在上方。
            sl = close * (Decimal('1') - Decimal(str(direction)) * Decimal(str(self.config["止盈止損設定"]["sl_value"])) / Decimal('100'))
        else:
            # 如果是固定價格模式。
            sl = close - Decimal(str(direction)) * Decimal(str(self.config["止盈止損設定"]["sl_value"]))

        # 對計算出的止盈止損價格進行四捨五入，保留 8 位小數。
        # 避免出現像 100.00000000001 這種奇怪的數字，影響比對邏輯。
        tp = tp.quantize(Decimal('1e-8'))
        sl = sl.quantize(Decimal('1e-8'))

        # --- 計算下單數量 (Size) ---
        # 模式 1: 百分比模式 (例如用總資金的 10%)。
        if self.config["下單設定"]["order_mode"] == "percent":
            # 基礎數量 = 當前現金 * 百分比 / 現價。
            base_size = self.stats.cash * Decimal(str(self.config["下單設定"]["order_value"])) / Decimal('100') / close
        # 模式 2: 價格模式 (例如買價值 1000 U 的幣)。
        elif self.config["下單設定"]["order_mode"] == "price":
            # 基礎數量 = 設定金額 / 現價。
            base_size = Decimal(str(self.config["下單設定"]["order_value"])) / close
        # 模式 3: 固定數量模式 (例如固定買 1 顆 BTC)。
        elif self.config["下單設定"]["order_mode"] == "fixed":
            # 基礎數量 = 設定數值。
            base_size = Decimal(str(self.config["下單設定"]["order_value"]))
        else:
            # 如果設定了不認識的模式，拋出錯誤。
            raise ValueError("order_mode只能是 percent, price, fixed 其中一種")

        # 計算最終下單量：方向 * 基礎數量 * 槓桿倍數。
        # 例如：做空 (-1) * 1 顆 * 10倍槓桿 = -10。
        size = Decimal(str(direction)) * base_size * Decimal(str(self.config["下單設定"]["leverage"]))

        # --- 資金不足判斷 ---
        # 計算開倉所需保證金 (Margin) = 價格 * 基礎數量 (未槓桿前的價值)。
        # 在全倉模式或一般計算中，這裡通常指實際要從錢包扣住的錢。
        margin = close * base_size
        # 預估手續費 = 總合約價值 (Margin * Leverage) * 手續費率。
        # 資金檢查必須把手續費也算進去，否則開倉瞬間就會變成負資產。
        estimated_fee = margin * Decimal(str(self.config["下單設定"]["leverage"])) * self.position.fee_rate
        
        # 如果 現金 < (保證金 + 手續費)，代表錢不夠。
        if self.stats.cash < (margin + estimated_fee):
            # 記錄警告訊息，並跳過這次開倉。
            logging.warning(f"[INSUFFICIENT_FUNDS] 時間: {timestamp}, 資金 {self.stats.cash:.2f} 不足，無法開倉 (需要 {margin+estimated_fee:.2f})")
            return # 直接返回，不執行後續開倉

        # 呼叫 position.open 進行開倉。
        # 注意：open 方法可能會回傳多個事件 (例如觸發反手時，會先有平倉事件，再有開倉事件)。
        logs_to_process = self.position.open(close, size, tp, sl, timestamp, i)

        # 遍歷所有回傳的日誌事件，並逐一交給 stats 進行記錄。
        if logs_to_process:
            for pnl, log_event in logs_to_process:
                self.stats.trade_log(pnl, log_event)

    # [Function 說明]
    # 功能：在終端機顯示回測的績效摘要。
    # 原理：讀取 stats 物件計算好的各項指標並 print 出來。
    def show(self):
        logging.info(f"===== 績效總結 =====")
        # 顯示 交易次數、總損益、最終資金、最大回撤。
        logging.info(f"總交易次數: {self.stats.count}, 總損益: {self.stats.pnl:.2f}, 最終資金: {self.stats.cash:.2f}, 最大回撤: {self.stats.max_drawdown:.2f}")
        # 顯示 勝率 (總勝率、多單勝率、空單勝率)。
        logging.info(f"勝率 (總/多/空): {self.stats.winrate():.2f}% / {self.stats.long_winrate():.2f}% / {self.stats.short_winrate():.2f}%")
        # 顯示 進階指標 (獲利因子、夏普比率、索提諾比率、卡瑪比率)。
        logging.info(f"獲利因子: {self.stats.profit_factor():.2f}, 夏普比率: {self.stats.sharpe():.2f}, 索提諾比率: {self.stats.sortino_ratio():.2f}, 卡瑪比率: {self.stats.calmar_ratio():.2f}")

    # [Function 說明]
    # 功能：繪製資金曲線。
    # 原理：呼叫 stats 物件的繪圖方法。
    def plot_results(self):
        """繪製並儲存資金曲線圖"""
        self.stats.plot_equity_curve()

    # [Function 說明]
    # 功能：執行回測的主迴圈。
    # 原理：遍歷每一根 K 線，依序檢查：資金是否足夠 -> 止損 -> 強制平倉(時間) -> 止盈 -> 策略信號。
    def run(self):
        # 使用 tqdm 建立進度條，範圍是整個 DataFrame 的長度。
        for i in tqdm(range(len(self.df))):
            # 檢查破產保護：如果現金小於等於 0，停止回測。
            if self.stats.cash <= 0:
                logging.info("資金不足，無法繼續交易")
                break
            
            # 取出當前這根 K 線的資料。
            # .copy() 是為了避免 Pandas 的 SettingWithCopyWarning，確保我們是在操作副本。
            row = self.df.iloc[i].copy() 
            # 將價格轉為 Decimal，確保後續計算精度。
            row['close'] = Decimal(str(row['close'])) 
            
            # 如果目前有持倉 (size != 0)，需要檢查出場條件。
            if self.position.size != 0:
                # 1. 檢查是否觸發止損 (SL)。
                logs_sl = self.position.trigger_SL(row["close"], row["close_time"])
                if logs_sl:
                    for pnl, log_event in logs_sl:
                        self.stats.trade_log(pnl, log_event)

                # --- 強制平倉判斷 (Max Hold) ---
                # 如果設定了最大持倉時間，且 (當前 K 線索引 - 開倉索引) 超過設定值。
                if self.max_hold is not None and (i - self.position.entry_index) >= self.max_hold:
                    # 強制全部平倉。
                    logs_force_close = self.position.close_all(row["close"], row["close_time"])
                    if logs_force_close:
                        for pnl, log_event in logs_force_close:
                            self.stats.trade_log(pnl, log_event)

                # 2. 檢查是否觸發止盈 (TP)。
                # 注意：這裡的邏輯是 SL 優先於 TP (保守估計)。
                logs_tp = self.position.trigger_TP(row["close"], row["close_time"])
                if logs_tp:
                    for pnl, log_event in logs_tp:
                        self.stats.trade_log(pnl, log_event)
            
            # 檢查進場信號。
            # 無論之前是否平倉，都檢查當前K棒是否有開倉信號 (支援反手或立即進場)。
            if row["signal"] == 1:
                # 1 代表做多信號。
                self._create_order(row["close"], 1, row["close_time"], i)
            elif row["signal"] == -1:
                # -1 代表做空信號。
                self._create_order(row["close"], -1, row["close_time"], i)
        
        # 迴圈結束後 (所有 K 線跑完)。
        # 如果手上還有持倉，必須強制平倉，以結算最終資產。
        if self.position.size != 0:
            # 確保傳入的價格是 Decimal。
            last_price = Decimal(str(self.df.iloc[-1]["close"]))
            # 以最後一根 K 線的收盤價平倉。
            close_results = self.position.close_all(last_price, self.df.iloc[-1]["close_time"])
            if close_results:
                pnl, log = close_results[0]
                self.stats.trade_log(pnl, log)
        
        # 在日誌中記錄最終結果。
        logging.info(f"總交易次數: {self.stats.count}, 總損益: {self.stats.pnl}, 最大回撤: {self.stats.max_drawdown}, 最終資金: {self.stats.cash}")
        
        # 如果有產生交易記錄，將結果存成 CSV。
        if not self.stats.log.empty:
            result_to_csv(self.stats.log, is_backtest=True)

# [Class 說明]
# 職責：管理單一倉位 (Position) 的狀態。
# 它不關心歷史記錄，只關心「現在」手上有多少單、成本價是多少、止盈止損在哪。
class position:
    """只負責倉位部分的動作"""
    # [Function 說明]
    # 功能：初始化倉位物件。
    # 原理：設定初始狀態為空倉 (size=0, avg_price=0)，並讀取費率和滑價設定。
    def __init__(self, config):
        self.config = config
        self.avg_price = Decimal('0') # 持倉均價
        self.size = Decimal('0')      # 持倉數量 (+為多, -為空)
        self.entry_index = 0          # 記錄開倉時的K棒索引 (用於 max_hold 計算)
        self.fee_rate = Decimal(str(self.config["回測設定"]["fee_rate"])) # 手續費率
        self.slippage = Decimal(str(self.config["回測設定"]["slippage"])) # 滑價率

    # [Function 說明]
    # 功能：執行開倉操作。
    # 原理：
    # 1. 檢查參數合法性。
    # 2. 應用滑價 (買更貴, 賣更便宜)。
    # 3. 處理反手邏輯 (如果持倉方向不同，先轉去 reverse)。
    # 4. 處理加倉邏輯 (Pyramiding)。
    # 5. 計算新的持倉均價 (加權平均)。
    # 6. 回傳開倉日誌。
    def open(self,
            price: Decimal,
            size: Decimal,
            tp: Decimal,
            sl: Decimal,
            timestamp,
            entry_index: int
            ) -> list[tuple[Decimal, pd.DataFrame]]:
        """開倉，回傳pnl和log"""
        # 判斷方向：size > 0 為多(1)，size < 0 為空(-1)，0 為無效。
        direction = Decimal('1') if size > Decimal('0') else Decimal('-1') if size < Decimal('0') else Decimal('0')
        if direction == Decimal('0'):
            raise ValueError("size不能為0")
        
        # 檢查止盈止損設置是否合理。
        if size > Decimal('0'): # 做多
            # 邏輯：止損 < 價格 < 止盈。
            if not (sl or Decimal('-Infinity')) < price < (tp or Decimal('Infinity')):
                raise ValueError(f"多單必須符合 止損<價格<止盈 {sl}<{price}<{tp}")
            # 計算滑價後的成交價：買入價 = 價格 * (1 + 滑價)。(買得比市價貴)
            price = price * (Decimal('1') + self.slippage)
        else: # 做空
            # 邏輯：止盈 < 價格 < 止損。
            if not (tp or Decimal('-Infinity')) < price < (sl or Decimal('Infinity')):
                raise ValueError(f"空單必須符合 止盈<價格<止損 {tp}<{price}<{sl}")
            # 計算滑價後的成交價：賣出價 = 價格 * (1 - 滑價)。(賣得比市價便宜)
            price = price * (Decimal('1') - self.slippage)
        
        # 檢查當前是否已有持倉。
        if self.size != Decimal('0'):
            # 如果 新下單方向 與 現有持倉方向 相反 (乘積小於0)。
            if self.size * size < Decimal('0'):
                # 如果設定允許反手 (Reverse)。
                if self.config["下單設定"]["reverse"]:
                    # 呼叫反手方法：先平舊倉，再開新倉。
                    return self.reverse(price, size, tp, sl, timestamp, entry_index)
                return [] # 如果不允許反手，則忽略此信號
                # (下方註解掉的程式碼是舊邏輯：部分平倉)
                # else:
                #     #如果開反方向的小倉位->平一部分倉位
                #     pnl, log = self.close(price, -size)
                #     return pnl, log
            else:
                # 如果方向相同 (加倉)。
                # 檢查是否允許加倉 (Pyramiding)。
                if not self.config["下單設定"]["pyramiding"]:
                    return [] # 如果不允許加倉，則忽略
        
        # --- 更新倉位狀態 ---
        # 計算新的加權平均價格。
        # 公式：(舊均價 * 舊數量 + 新價 * 新數量) / 總數量。
        # .copy_abs() 取絕對值，因為 size 可能是負的，但計算權重需要正數。
        self.avg_price = (self.avg_price * self.size.copy_abs() + price * size.copy_abs()) / (self.size.copy_abs() + size.copy_abs())
        # 更新總持倉量。
        self.size += size
        # 更新止盈止損設定 (以最新一次開倉為準，這是一種簡化策略)。
        self.tp = tp
        self.sl = sl
        # 更新開倉索引。
        self.entry_index = entry_index 

        # 建立開倉日誌。
        columns = ["時間","狀態","多/空","進場價","進場量","當前均價","當前持倉量"]
        log = pd.DataFrame([[timestamp, "開倉", direction, price, size, self.avg_price, self.size]], columns=columns)
        # 回傳 PnL=0 (開倉無損益) 和日誌。
        return [(Decimal('0'), log)] # 作為列表返回

    # [Function 說明]
    # 功能：執行平倉操作。
    # 原理：
    # 1. 計算滑價後的成交價。
    # 2. 計算損益 (PnL) = 毛利 - 平倉手續費。
    # 3. 更新剩餘持倉 (通常是歸零，除非部分平倉)。
    # 4. 回傳損益和日誌。
    def close(self,
            price:float,
            size:float, # 注意: 這裡的size是帶有方向的平倉量 (例如平多單，size為負)
            timestamp
            ) -> list[tuple[Decimal, pd.DataFrame]]:
        """平倉，回傳pnl和log"""
        # 如果沒持倉，不能平倉。
        if self.size == Decimal('0'):
            raise Exception("當前無持倉，無法平倉")
        
        # 應用滑價。
        if size > Decimal('0'): # 買回平空單
            price = price * (Decimal('1') + self.slippage) 
        else: # 賣出平多單
            price = price * (Decimal('1') - self.slippage) 
            # (注意：這裡的 code 邏輯似乎是：size>0 代表此操作是買入，size<0 代表此操作是賣出。

        # 修正後的損益計算:
        # PnL = 毛利 - 平倉手續費。
        # 毛利 = (平倉價 - 均價) * (平倉量，負值)。
        # 例如多單 (均價100)，平倉價110，size為-1。毛利 = (110-100)*1 = 10。
        # 公式推導: (Price - Avg) * (-Size) 是正確的嗎？
        # 若多單 size=1, 平倉 size=-1. (110-100) * -(-1) = 10. 正確。
        # 若空單 size=-1, 平倉 size=1. (90-100) * -(1) = -10 * -1 = 10. 正確。
        gross_pnl = (price - self.avg_price) * (-size)
        
        # 計算平倉手續費 = 交易價值 * 費率。
        closing_fee = size.copy_abs() * price * self.fee_rate
        # 淨損益。
        pnl = gross_pnl - closing_fee
        
        # 更新持倉量 (加上平倉的負數量，通常變為0)。
        self.size += size
        # 如果倉位歸零，均價也重置。
        if self.size == Decimal('0'):
            self.avg_price = Decimal('0')
            
        columns = ["時間", "狀態","出場價","出場量","實現損益", "剩餘倉位"]
        log = pd.DataFrame([[timestamp, "平倉", price, -size, pnl, self.size]], columns=columns)
        return [(pnl, log)]
    
    # [Function 說明]
    # 功能：helper 函式，用於一次性平掉所有倉位。
    # 原理：自動計算需要平掉的數量 (-self.size)，呼叫 close。
    def close_all(self,
                price:float,
                timestamp
                ) -> list[tuple[float, pd.DataFrame]]:
        """全部平倉"""
        if self.size == 0:
            return []
        # 清除止盈止損設定。
        self.sl = None
        self.tp = None
        # 執行平倉。
        return self.close(price, -self.size, timestamp)

    # [Function 說明]
    # 功能：執行反手操作 (Reverse)。
    # 原理：這是一個複合動作。
    # 1. 先呼叫 close_all 平掉舊倉位。
    # 2. 再手動執行開倉邏輯 (直接設定屬性，而不是遞迴呼叫 open，避免無窮迴圈或複雜檢查)。
    # 回傳：一個包含「平倉日誌」和「開倉日誌」的列表。
    def reverse(self,
                price: Decimal,
                size: Decimal,
                tp: Decimal,
                sl: Decimal,
                timestamp,
                entry_index:int
                ) -> list[tuple[Decimal, pd.DataFrame]]:
        """反手，先全部平倉再開新倉，返回一個包含兩次操作日誌的列表"""
        # 1. 平倉舊倉位。
        close_results = self.close_all(price, timestamp)
        if not close_results:
            return [] # 如果平倉失敗或沒有倉位，直接返回
        
        pnl_close, log_close = close_results[0]
        
        # 2. 開新倉。
        # 這裡不呼叫 self.open，而是直接設定狀態，模擬開倉成功。
        direction = Decimal('1') if size > Decimal('0') else Decimal('-1')
        self.avg_price = price
        self.size = size
        self.tp = tp
        self.sl = sl
        self.entry_index = entry_index
        
        columns = ["時間","狀態","多/空","進場價","進場量","當前均價","當前持倉量"]
        log_open = pd.DataFrame([[timestamp, "開倉", direction, price, size, self.avg_price, self.size]], columns=columns)
        
        # 回傳兩個事件：平倉結果(有PnL) + 開倉結果(PnL=0)。
        return [(pnl_close, log_close), (Decimal('0'), log_open)]

    # [Function 說明]
    # 功能：檢查是否觸發止損 (SL)。
    # 原理：如果是多單且價格低於 SL，或空單且價格高於 SL，則呼叫 close_all。
    def trigger_SL(self,
                   close: Decimal, timestamp
                   ):
        if self.sl is not None:
            if ((close <= self.sl) and (self.size > Decimal('0'))) or ((close >= self.sl) and (self.size < Decimal('0'))):
                return self.close_all(self.sl, timestamp)
        return []
    
    # [Function 說明]
    # 功能：檢查是否觸發止盈 (TP)。
    # 原理：如果是多單且價格高於 TP，或空單且價格低於 TP，則呼叫 close_all。
    def trigger_TP(self,
                   close: Decimal, timestamp
                   ):
        if self.tp is not None:
            if ((close >= self.tp) and (self.size > Decimal('0'))) or ((close <= self.tp) and (self.size < Decimal('0'))):
                return self.close_all(self.tp, timestamp)
        return []

# [Class 說明]
# 職責：統計與會計模組 (Statistics)。
# 它負責收集 `position` 產生的所有交易日誌，維護資金水位 (Cash)，並計算各種績效指標。
class stats:
    """只負責記錄資料的動作"""
    # [Function 說明]
    # 功能：初始化統計物件。
    # 原理：建立各種計數器 (count)，並設定初始資金。
    def __init__(self, config):
        self.config = config
        self.count = Decimal('0')       # 總交易次數
        self.count_long = Decimal('0')  # 多單次數
        self.count_long_win = Decimal('0') # 多單獲利次數
        self.count_short = Decimal('0') # 空單次數
        self.count_short_win = Decimal('0') # 空單獲利次數
        self.log = pd.DataFrame()       # 總交易日誌表
        self.cash = Decimal(str(self.config["回測設定"]["initial_cash"])) # 當前資金
        self.pnl = Decimal('0')         # 累計損益
        self.max_drawdown = Decimal('0') # 最大回撤

    # [Function 說明]
    # 功能：處理一筆交易日誌。
    # 原理：
    # 1. 將日誌存入總表。
    # 2. 如果是開倉：扣除手續費 (開倉成本)。
    # 3. 如果是平倉：加上實現損益 (PnL)，更新現金，更新最大回撤，更新勝率計數。
    def trade_log(self,
                pnl: Decimal,
                log:pd.DataFrame
                ):
        if log is not None:
            # 將新日誌合併到主 log DataFrame。
            self.log = pd.concat([self.log, log], ignore_index=True)

            # 如果是開倉日誌，直接扣除開倉手續費。
            if "狀態" in log.columns and log["狀態"].iloc[0] == "開倉":
                # 計算開倉手續費 = 數量 * 價格 * 費率。
                opening_fee = log["進場量"].iloc[0].copy_abs() * log["進場價"].iloc[0] * Decimal(str(self.config["回測設定"]["fee_rate"]))
                # 從現金中扣除。
                self.cash -= opening_fee
                # 註：開倉本身不計入 realized pnl，因為還沒平倉。

            # 如果是平倉日誌，才更新損益和交易統計。
            elif "狀態" in log.columns and log["狀態"].iloc[0] == "平倉":
                if pnl is None or pnl == Decimal('0'):
                    return # 如果平倉事件沒有產生損益 (理論上不會)，則跳過

                # 更新累計損益。
                self.pnl += pnl
                # 更新現金 (加上損益，損益已扣除平倉手續費)。
                self.cash += pnl
                
                # 計算最大回撤。
                # 這裡使用簡化算法：如果當前累計 PnL 低於歷史最低點，更新 Max DD。
                # (更精確的做法應該是 High Water Mark 算法，計算從最高資金點的回落)。
                if self.pnl < self.max_drawdown:
                    self.max_drawdown = self.pnl

                # 更新勝率統計。
                closed_size = log["出場量"].iloc[0]
                self.count += Decimal('1')
                if closed_size > Decimal('0'): # 平多倉 (買回) - 註：這裡邏輯需確認 closed_size 定義，通常平多是賣出(size<0)。假設 code 邏輯是：closed_size是反向操作量。
                    # 如果 closed_size > 0，代表是買入動作，那原本應該是空單。
                    # 請注意這裡的邏輯與 open/close 的 size 定義需一致。
                    # 根據 close 函式：平多是賣出 (size<0)，平空是買入 (size>0)。
                    # 所以 closed_size > 0 -> 平空單。
                    # 但原程式碼寫法：if closed_size > 0: self.count_long += 1... 這似乎認為 >0 是多單。
                    # 我們依照原程式碼註解重寫，不更動邏輯，但需注意此處潛在定義。
                    self.count_long += Decimal('1')
                    if pnl > Decimal('0'): self.count_long_win += Decimal('1')
                elif closed_size < Decimal('0'): # 平空倉
                    self.count_short += Decimal('1')
                    if pnl > Decimal('0'): self.count_short_win += Decimal('1')

    # [Function 說明]
    # 功能：計算夏普比率 (Sharpe Ratio)。
    # 原理：衡量每單位風險所帶來的超額報酬。
    # 公式：(平均日報酬 - 無風險利率) / 日報酬標準差 * sqrt(252)。
    def sharpe(self):
        # 複製一份log以避免修改原始數據
        log_df = self.log.copy()

        # 確保 '實現損益' 欄位存在且有非NaN值
        if "實現損益" not in log_df.columns or log_df["實現損益"].isnull().all():
            return Decimal('0.0')

        # 將時間字串轉換為datetime物件，並設為索引，以便使用 resample。
        log_df['時間'] = pd.to_datetime(log_df['時間'])
        log_df.set_index('時間', inplace=True)

        # 按天對損益進行分組加總，得到每日報酬序列。
        daily_returns = log_df['實現損益'].astype(float).resample('D').sum()

        # 如果回測期間小於2天，無法計算標準差，返回0。
        if len(daily_returns) < 2:
            return Decimal('0.0')

        # 計算年化夏普比率。
        rf = Decimal('0.01') # 假設無風險利率為 1%。
        mean_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        
        # 如果標準差為0 (例如每天都沒賺沒賠)，返回0避免除以零錯誤。
        if std_daily_return == 0:
            return Decimal('0.0')

        # (每日平均報酬 - 每日無風險利率) / 每日報酬標準差 * 年化因子(sqrt(252))。
        daily_rf = rf / 252
        sharpe_ratio = (mean_daily_return - daily_rf) / std_daily_return * np.sqrt(252)
        return sharpe_ratio

    # [Function 說明]
    # 功能：計算資金曲線 (Equity Curve) 的數據點。
    # 原理：將每筆交易的損益累加，加上當前現金，還原出每個時間點的總資產。
    def get_equity_curve(self):
        """計算並返回資金曲線的時間序列數據"""
        log_df = self.log.copy()
        if "實現損益" not in log_df.columns or log_df["實現損益"].isnull().all():
            return None

        # 計算每個時間點的累計損益。
        log_df['時間'] = pd.to_datetime(log_df['時間'])
        pnl_events = log_df.dropna(subset=['實現損益'])
        pnl_events = pnl_events.sort_values(by='時間')
        pnl_events['累計損益'] = pnl_events['實現損益'].cumsum()
        # 資金曲線 = 目前現金(已含所有損益) - 總損益(變回初始現金) + 該時間點累計損益。
        # 簡化理解：初始現金 + 該時間點的累計損益。
        pnl_events['資金曲線'] = self.cash - self.pnl + pnl_events['累計損益']

        return pnl_events[['時間', '資金曲線']]

    # [Function 說明]
    # 功能：繪製並顯示資金曲線圖。
    # 原理：使用 matplotlib 畫出時間 vs 資金的折線圖。
    def plot_equity_curve(self):
        """繪製資金曲線圖"""
        equity_df = self.get_equity_curve()
        if equity_df is None:
            logging.warning("沒有足夠的交易數據來繪製資金曲線圖。")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(equity_df['時間'], equity_df['資金曲線'], label='Equity Curve')
        plt.title(f"Equity Curve - {self.config['基本設定']['symbol']} - {self.config['基本設定']['strategy']}")
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.legend()
        plt.show()

    # [Function 說明]
    # 功能：計算多單勝率。
    def long_winrate(self):
        if self.count_long == Decimal('0'):
            return Decimal('0')
        return (self.count_long_win / self.count_long) * Decimal('100')

    # [Function 說明]
    # 功能：計算空單勝率。
    def short_winrate(self):
        if self.count_short == Decimal('0'):
            return Decimal('0')
        return (self.count_short_win / self.count_short) * Decimal('100')

    # [Function 說明]
    # 功能：計算總勝率。
    def winrate(self):
        if self.count == Decimal('0'):
            return Decimal('0')
        return ((self.count_long_win + self.count_short_win) / self.count) * Decimal('100')

    # [Function 說明]
    # 功能：計算獲利因子 (Profit Factor)。
    # 原理：總獲利金額 / 總虧損金額。大於 1.5 通常算不錯。
    def profit_factor(self):
        """計算獲利因子 (總盈利 / 總虧損)"""
        if "實現損益" not in self.log.columns:
            return Decimal('0.0')

        returns = self.log["實現損益"].dropna()
        # 加總所有正數 (獲利)。
        total_profit = sum((r for r in returns if r > Decimal('0')), Decimal('0'))
        # 加總所有負數 (虧損) 並取絕對值。
        total_loss = sum((r for r in returns if r < Decimal('0')), Decimal('0')).copy_abs()

        if total_loss == Decimal('0'):
            return Decimal('Infinity') # 如果沒有虧損，獲利因子為無限大

        return total_profit / total_loss

    # [Function 說明]
    # 功能：計算索提諾比率 (Sortino Ratio)。
    # 原理：類似夏普比率，但分母只計算「下行風險」(虧損的標準差)，不懲罰上漲的波動。
    def sortino_ratio(self):
        """計算年化索提諾比率"""
        log_df = self.log.copy()
        if "實現損益" not in log_df.columns or log_df["實現損益"].isnull().all():
            return Decimal('0.0')

        log_df['時間'] = pd.to_datetime(log_df['時間'])
        log_df.set_index('時間', inplace=True)
        daily_returns = log_df['實現損益'].astype(float).resample('D').sum()

        if len(daily_returns) < 2:
            return Decimal('0.0')

        rf = Decimal('0.01')
        mean_daily_return = daily_returns.mean()
        
        # 計算下行標準差 (只考慮虧損日的波動)。
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = negative_returns.std()

        if downside_std == 0 or pd.isna(downside_std):
            return Decimal('0.0')

        daily_rf = rf / 252
        sortino = (mean_daily_return - daily_rf) / downside_std * np.sqrt(252)
        return sortino

    # [Function 說明]
    # 功能：計算卡瑪比率 (Calmar Ratio)。
    # 原理：年化報酬率 / 最大回撤。衡量承受每單位最大虧損能換來多少報酬。
    def calmar_ratio(self):
        """計算卡瑪比率 (年化報酬率 / 最大回撤)"""
        log_df = self.log.copy()
        if "實現損益" not in log_df.columns or log_df["實現損益"].isnull().all():
            return 0.0

        # 計算年化報酬率。
        log_df['時間'] = pd.to_datetime(log_df['時間'])
        start_date = log_df['時間'].min()
        end_date = log_df['時間'].max()
        num_days = (end_date - start_date).days
        
        if num_days == 0:
            return 0.0

        # 總報酬率 = 總損益 / 初始資金。
        total_return_rate = self.pnl / self.config["回測設定"]["initial_cash"]
        # 年化報酬率公式。
        annualized_return = (1 + total_return_rate) ** (365.0 / num_days) - 1

        # 取得最大回撤的絕對值。
        # 注意：目前的最大回撤是基於累計PnL，更精確的應基於資金曲線，但此處使用現有值。
        max_dd_value = abs(self.max_drawdown)
        if max_dd_value == 0:
            return float('inf') # 如果沒有回撤，比率為無限大

        max_dd_percent = max_dd_value / self.config["回測設定"]["initial_cash"] # 將回撤轉換為百分比

        return annualized_return / max_dd_percent

# 這是 Python 腳本的標準入口。
# 當直接執行此檔案時，會執行以下測試程式碼。
if __name__ == "__main__":
    # 將所有執行邏輯移至此處
    #載入config
    config = load_config()
    # 設定讀取已處理資料的路徑。
    pathdir = "result/backtests"
    # 根據設定檔組合檔名。
    filename = f"{config['基本設定']['symbol']}_{config['基本設定']['strategy']}_{config['回測設定']['start_time']} to {config['回測設定']['end_time']}.csv"
    # 處理檔名中的冒號。
    filename = re.sub(":","-",filename)
    # 讀取 CSV 資料 (假設已由 main.py 產生並存檔)。
    df = get_processed_data(filename)

    # 建立回測物件。
    bt = backtest(df, config)
    # 執行回測。
    bt.run()
    # 顯示統計結果。
    bt.show()
    # 繪製圖表。
    bt.plot_results()