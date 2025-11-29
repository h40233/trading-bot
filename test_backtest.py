import pytest
import pandas as pd
import numpy as np
import math
from decimal import Decimal
import logging
from unittest.mock import patch
from backtest import position, stats, backtest

@pytest.fixture
def sample_df():
    """提供一個用於回測的樣本 DataFrame"""
    data = {
        # 時間需要是 datetime 物件，以便 resample
        'close_time': pd.to_datetime(['2024-01-01 01:00', '2024-01-01 02:00', '2024-01-01 03:00', '2024-01-01 04:00']),
        'close': [100, 110, 95, 105],
        'signal': [1, 0, -1, 0]  # 買入 -> 持有 -> 反手做空 -> 持有
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_config():
    """模擬 config，避免測試依賴於真實的 config.json 檔案"""
    config_data = {
        "基本設定": {"symbol": "TESTUSDT", "strategy": "TEST", "max_hold": None}, # 預設不啟用 max_hold
        "下單設定": {"order_mode": "fixed", "order_value": 1, "leverage": 1, "pyramiding": False, "reverse": True},
        "止盈止損設定": {"tp_of_percent": True, "tp_value": 10, "sl_of_percent": True, "sl_value": 5},
        "回測設定": {"initial_cash": 10000, "fee_rate": 0.001, "slippage": 0.0} # 測試時滑價設為0，方便計算
    }
    return config_data

class TestPosition:
    """測試 position class 的核心功能"""
    def test_open_long(self, mock_config):
        pos = position(mock_config)
        # open 現在回傳一個列表
        results = pos.open(price=Decimal('100'), size=Decimal('1'), tp=Decimal('110'), sl=Decimal('95'), timestamp='2024-01-01 01:00', entry_index=0)
        pnl, log = results[0]
        assert pnl == Decimal('0')
        assert pos.size == Decimal('1')
        assert pos.avg_price == Decimal('100')
        assert "開倉" in log["狀態"].values

    def test_close_long_profit(self, mock_config):
        pos = position(mock_config)
        pos.open(price=Decimal('100'), size=Decimal('1'), tp=Decimal('110'), sl=Decimal('95'), timestamp='2024-01-01 01:00', entry_index=0)
        
        # 在 105 元平倉
        results = pos.close(price=Decimal('105'), size=Decimal('-1'), timestamp='2024-01-01 02:00')
        pnl, log = results[0]
        
        # 新邏輯: PnL = 毛利 - 平倉手續費
        # PnL = (105 - 100) * 1 - (105 * 0.001) = 5 - 0.105 = 4.895
        assert math.isclose(pnl, Decimal('4.895'))
        assert pos.size == Decimal('0')
        assert "平倉" in log["狀態"].values

    def test_reverse_to_short(self, mock_config):
        pos = position(mock_config)
        pos.open(price=Decimal('100'), size=Decimal('1'), tp=Decimal('110'), sl=Decimal('95'), timestamp='2024-01-01 01:00', entry_index=0)
        
        # 由於 sl=95，當價格到達95時，會觸發止損，所以平倉價是95
        # reverse 現在回傳一個包含平倉和開倉日誌的列表
        results = pos.reverse(price=Decimal('95'), size=Decimal('-1'), tp=Decimal('85.5'), sl=Decimal('99.75'), timestamp='2024-01-01 03:00', entry_index=1)
        
        # 驗證平倉部分的損益
        pnl_close, log_close = results[0]
        
        # 新邏輯: PnL = 毛利 - 平倉手續費
        # PnL = (95 - 100) * 1 - (95 * 0.001) = -5 - 0.095 = -5.095
        assert math.isclose(pnl_close, Decimal('-5.095'))
        assert pos.size == Decimal('-1') # 倉位變成 -1 (空倉)
        assert math.isclose(pos.avg_price, Decimal('95')) # 均價更新為 95

    def test_pyramiding(self, mock_config):
        """測試加倉功能"""
        mock_config["下單設定"]["pyramiding"] = True
        pos = position(mock_config)

        # 第一次開倉
        pos.open(price=Decimal('100'), size=Decimal('1'), tp=Decimal('110'), sl=Decimal('95'), timestamp='2024-01-01 01:00', entry_index=0)
        assert pos.size == Decimal('1')
        assert pos.avg_price == Decimal('100')

        # 第二次加倉
        pos.open(price=Decimal('110'), size=Decimal('1'), tp=Decimal('120'), sl=Decimal('100'), timestamp='2024-01-01 02:00', entry_index=1)
        assert pos.size == Decimal('2') # 驗證持倉量增加
        assert math.isclose(pos.avg_price, Decimal('105')) # 驗證均價被重新計算 (100*1 + 110*1) / 2

        # 第三次部分平倉 (平掉一半)
        results = pos.close(price=Decimal('120'), size=Decimal('-1'), timestamp='2024-01-01 03:00')
        pnl, log = results[0]
        assert pos.size == Decimal('1') # 驗證剩餘持倉
        # 新邏輯: PnL = 毛利 - 平倉手續費
        # PnL = (120 - 105) * 1 - (120 * 0.001) = 15 - 0.12 = 14.88
        assert math.isclose(pnl, Decimal('14.88'))

class TestStats:
    """測試 stats class 的計算功能"""
    def test_trade_log_and_metrics(self, mock_config):
        stat = stats(mock_config)
        
        # 模擬一筆獲利的多單交易
        profit_log = pd.DataFrame([{'狀態': '平倉', '出場量': Decimal('1'), '實現損益': Decimal('100'), '時間': '2024-01-01'}])
        stat.trade_log(pnl=Decimal('100'), log=profit_log)
        
        # 模擬一筆虧損的空單交易
        loss_log = pd.DataFrame([{'狀態': '平倉', '出場量': Decimal('-1'), '實現損益': Decimal('-50'), '時間': '2024-01-02'}])
        stat.trade_log(pnl=Decimal('-50'), log=loss_log)
        
        assert stat.count == 2
        assert stat.count_long == 1
        assert stat.count_long_win == 1
        assert stat.count_short == 1
        assert stat.count_short_win == 0
        assert math.isclose(stat.pnl, Decimal('50'))
        assert math.isclose(stat.cash, Decimal('10050'))
        assert math.isclose(stat.winrate(), Decimal('50.0'))
        assert math.isclose(stat.profit_factor(), Decimal('2.0')) # 獲利因子 = 100 / 50 = 2.0

    def test_max_drawdown(self, mock_config):
        """測試最大回撤計算的正確性"""
        stat = stats(mock_config)
        # 模擬一系列交易來創造回撤
        stat.trade_log(pnl=Decimal('100'), log=pd.DataFrame([{'狀態': '平倉', '實現損益': Decimal('100'), '出場量': Decimal('1')}])) # pnl=100, max_dd=0
        assert math.isclose(stat.max_drawdown, Decimal('0'))

        stat.trade_log(pnl=Decimal('-200'), log=pd.DataFrame([{'狀態': '平倉', '實現損益': Decimal('-200'), '出場量': Decimal('1')}])) # pnl=-100, max_dd=-100
        assert math.isclose(stat.max_drawdown, Decimal('-100'))

        stat.trade_log(pnl=Decimal('50'), log=pd.DataFrame([{'狀態': '平倉', '實現損益': Decimal('50'), '出場量': Decimal('1')}])) # pnl=-50, max_dd=-100
        assert math.isclose(stat.max_drawdown, Decimal('-100'))

        stat.trade_log(pnl=Decimal('-100'), log=pd.DataFrame([{'狀態': '平倉', '實現損益': Decimal('-100'), '出場量': Decimal('1')}])) # pnl=-150, max_dd=-150
        assert math.isclose(stat.max_drawdown, Decimal('-150'))

@patch('backtest.result_to_csv') # 模擬檔案寫入
@patch('backtest.plt.show')     # 模擬繪圖顯示
def test_backtest_run_integration(mock_plt_show, mock_result_to_csv, mock_config, sample_df):
    """對 backtest.run() 進行整合測試，驗證完整流程"""
    bt = backtest(sample_df,mock_config)
    bt.run()

    # --- 驗證最終結果 ---
    # 新邏輯: PnL 是平倉損益的總和，開倉費直接從現金扣除
    # 交易1 (止盈): PnL = (110-100)*1 - (110*0.001) = 9.89
    # 交易2 (止損): PnL = (95-99.75)*1 - (99.75*0.001) = -4.75 - 0.09975 = -4.84975
    # 總 PnL = 9.89 - 4.84975 = 5.04025
    # 最終資金 = 初始資金 + 總PnL - 總開倉費 = 10000 + 5.04025 - (100*0.001) - (95*0.001) = 10000 + 5.04025 - 0.1 - 0.095 = 10004.84525
    # 交易次數應為 2 (一次止盈，一次止損)
    assert bt.stats.count == 2
    assert math.isclose(bt.stats.pnl, Decimal('5.04025'))
    assert math.isclose(bt.stats.cash, Decimal('10004.84525'))

    # 驗證 mock 函式被呼叫
    mock_result_to_csv.assert_called_once()
    # 注意: plot_results 是在 __main__ 區塊呼叫的，所以 run() 內部不會呼叫到 plt.show()
    # 如果要測試 plot，需要另外呼叫 bt.plot_results()
    # mock_plt_show.assert_called_once()

def test_insufficient_funds(mock_config, sample_df, caplog):
    """測試當資金不足時，是否會正確地阻止開倉"""
    # 將初始資金設定為一個極低的值，不足以支付保證金和手續費
    # 開倉在價格100，數量1，需要保證金 100*1=100，手續費 100*1*0.001=0.1。總共需要 100.1
    mock_config["回測設定"]["initial_cash"] = "50"
    
    bt = backtest(sample_df, mock_config)
    
    with caplog.at_level(logging.WARNING):
        bt.run()

    # 驗證：沒有任何交易發生
    assert bt.stats.count == 0
    # 驗證：最終資金仍然是初始資金 (因為沒有交易)
    assert math.isclose(bt.stats.cash, Decimal('50'))
    # 驗證：日誌中出現了資金不足的警告
    # 使用更健壯的方式檢查日誌記錄，避免因編碼問題導致的斷言失敗
    assert len(caplog.records) > 0
    assert any("[INSUFFICIENT_FUNDS]" in record.message for record in caplog.records)

def test_force_close_by_max_hold(mock_config):
    """測試 max_hold 強制平倉功能是否正常運作"""
    # 準備一個特殊的 DataFrame，只有開倉信號，後面都是持有
    data = {
        'close_time': pd.to_datetime(['2024-01-01 01:00', '2024-01-01 02:00', '2024-01-01 03:00', '2024-01-01 04:00']),
        'close': [Decimal('100'), Decimal('105'), Decimal('110'), Decimal('115')],
        'signal': [1, 0, 0, 0] 
    }
    df = pd.DataFrame(data)

    # 在設定中加入 max_hold=2
    # 預期行為：在 index=0 開倉，持有 index=1，在 index=2 時因為 (2-0)>=2，觸發強制平倉
    mock_config["回測設定"]["max_hold"] = 2
    
    bt = backtest(df, mock_config)
    bt.run()

    # 驗證：只發生了一筆交易 (強制平倉)
    assert bt.stats.count == 1
    # 驗證：最終倉位為 0
    assert bt.position.size == 0
    # 新邏輯: 驗證損益計算是否正確 (在價格110時平倉)
    # PnL = (110 - 100)*1 - (110*0.001) = 10 - 0.11 = 9.89
    assert math.isclose(bt.stats.pnl, Decimal('9.89'))

def test_leverage_trading(mock_config):
    """測試槓桿交易是否正確放大倉位"""
    mock_config["下單設定"]["leverage"] = 3
    mock_config["下單設定"]["order_value"] = 1 # 固定手數
    
    pos = position(mock_config)
    # 使用 patch 來模擬 position.open，因為我們只想測試 _create_order 的計算邏輯
    with patch.object(pos, 'open') as mock_open:
        # 設定 mock 物件的回傳值為一個列表，以符合新的格式
        mock_open.return_value = [(0, pd.DataFrame())]
        bt = backtest(pd.DataFrame(), mock_config)
        bt.position = pos # 將 mock 過的 position 物件賦給 backtest 實例
        bt._create_order(close=Decimal('100'), direction=1, timestamp='2024-01-01', i=0)

        # 驗證 position.open 被呼叫時，傳入的 size 是否被槓桿放大了
        # 預期 size = direction * base_size * leverage = 1 * 1 * 3 = 3
        mock_open.assert_called_once()
        called_args, _ = mock_open.call_args
        assert math.isclose(called_args[1], Decimal('3')) # 驗證 size 參數

def test_order_mode_percent(mock_config):
    """測試百分比開倉模式的下單量計算"""
    mock_config["下單設定"]["order_mode"] = "percent"
    mock_config["下單設定"]["order_value"] = 10 # 使用 10% 的資金
    
    bt = backtest(pd.DataFrame(), mock_config) # 初始資金為 10000
    # 預期 base_size = cash * percent / 100 / price = 10000 * 10 / 100 / 100 = 10
    bt._create_order(close=Decimal('100'), direction=1, timestamp='2024-01-01', i=0)
    assert math.isclose(bt.position.size, Decimal('10'))

class TestEdgeCases:
    """專門測試邊界條件和例外情況"""

    def test_empty_dataframe(self, mock_config):
        """測試當傳入空的 DataFrame 時，系統是否能正常處理"""
        bt = backtest(pd.DataFrame(), mock_config)
        # 期望 run() 能夠正常結束，不拋出任何錯誤
        try:
            bt.run()
        except Exception as e:
            pytest.fail(f"當傳入空 DataFrame 時，bt.run() 拋出了未預期的錯誤: {e}")
        # 驗證沒有任何交易發生
        assert bt.stats.count == 0
        assert bt.stats.pnl == Decimal('0')

    def test_close_non_existent_position(self, mock_config):
        """測試對一個不存在的倉位執行平倉時，是否會拋出例外"""
        pos = position(mock_config)
        assert pos.size == Decimal('0')
        # 使用 pytest.raises 來驗證是否拋出了預期的 Exception
        with pytest.raises(Exception, match="當前無持倉，無法平倉"):
            pos.close(price=Decimal('100'), size=Decimal('-1'), timestamp='2024-01-01')

    def test_zero_order_value(self, mock_config):
        """測試當 order_value 為 0 時，是否會阻止開倉並拋出例外"""
        mock_config["下單設定"]["order_value"] = 0
        bt = backtest(pd.DataFrame(), mock_config)
        # 預期在 _create_order -> position.open 中會因為 size 為 0 而拋出 ValueError
        with pytest.raises(ValueError, match="size不能為0"):
            bt._create_order(close=Decimal('100'), direction=1, timestamp='2024-01-01', i=0)

    def test_division_by_zero_on_price(self, mock_config):
        """測試當價格為 0 時，百分比或固定價值開倉是否會拋出例外"""
        mock_config["下單設定"]["order_mode"] = "percent"
        bt = backtest(pd.DataFrame(), mock_config)
        # Decimal 預設會捕捉除以零的錯誤並拋出 InvalidOperation
        from decimal import DivisionByZero
        with pytest.raises(DivisionByZero):
            bt._create_order(close=Decimal('0'), direction=1, timestamp='2024-01-01', i=0)

    def test_no_loss_profit_factor(self, mock_config):
        """測試在沒有任何虧損的情況下，獲利因子是否為無限大"""
        stat = stats(mock_config)
        profit_log = pd.DataFrame([{'狀態': '平倉', '出場量': Decimal('1'), '實現損益': Decimal('100'), '時間': '2024-01-01'}])
        stat.trade_log(pnl=Decimal('100'), log=profit_log)
        assert stat.profit_factor() == Decimal('Infinity')

    def test_consecutive_reverse_signals(self, mock_config):
        """測試連續的反手信號是否能被正確處理"""
        data = {
            'close_time': pd.to_datetime(['2024-01-01 01:00', '2024-01-01 02:00']),
            'close': [100, 105],
            'signal': [1, -1] 
        }
        df = pd.DataFrame(data)
        bt = backtest(df, mock_config)
        bt.run()

        # 預期：
        # 1. K棒i=1，反手操作，平掉多倉。PnL = (105-100) - 105*0.001 = 4.895。count=1。
        # 2. 反手後，持有空倉。
        # 3. 回測結束，使用最後價格105，強制平掉空倉。PnL = (105-105) - 105*0.001 = -0.105。count=2。
        # 總 PnL = 4.895 - 0.105 = 4.79
        assert bt.stats.count == 2 # 包含反手時的平倉，和期末的平倉
        assert bt.position.size == Decimal('0') # 最終倉位應為 0
        assert math.isclose(bt.stats.pnl, Decimal('4.79'))