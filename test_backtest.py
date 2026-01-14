# -----------------------------------------------------------------------------------------
# [檔案功能說明]
# 這是一個完整的測試套件，用於驗證 `backtest.py` 模組的所有功能。
# 測試範圍包括：
# 1. Position Class: 開倉、平倉、反手、加倉 (Pyramiding)、損益計算。
# 2. Stats Class: 交易紀錄統計、勝率、獲利因子、最大回撤 (Max Drawdown)。
# 3. Backtest Class (整合測試): 模擬完整的回測流程，驗證最終資金與交易次數。
# 4. Edge Cases: 測試資金不足、空資料、錯誤參數等異常狀況的處理能力。
# -----------------------------------------------------------------------------------------

import pytest
import pandas as pd
import numpy as np
import logging
from unittest.mock import patch, MagicMock
from backtest import position, stats, backtest

# [Fixture 說明]
@pytest.fixture
def sample_df():
    """提供一個用於回測的樣本 DataFrame"""
    data = {
        'close_time': pd.to_datetime(['2024-01-01 01:00', '2024-01-01 02:00', '2024-01-01 03:00', '2024-01-01 04:00']),
        'close': [100.0, 110.0, 95.0, 105.0], 
        'signal': [1, 0, -1, 0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_config():
    """模擬 config，避免測試依賴於真實的 config.json 檔案"""
    return {
        "基本設定": {"symbol": "TESTUSDT", "strategy": "TEST", "max_hold": None},
        "下單設定": {"order_mode": "fixed", "order_value": 1, "leverage": 1, "pyramiding": False, "reverse": True},
        "止盈止損設定": {"tp_of_percent": True, "tp_value": 10.0, "sl_of_percent": True, "sl_value": 5.0},
        "回測設定": {"initial_cash": 10000.0, "fee_rate": 0.001, "slippage": 0.0}
    }

# [Class 說明]
class TestPosition:
    """測試 position class 的核心功能"""
    
    def test_open_long(self, mock_config):
        pos = position(mock_config)
        results = pos.open(price=100.0, size=1.0, tp=110.0, sl=95.0, timestamp='2024-01-01 01:00', entry_index=0)
        pnl, log = results[0]
        
        assert pnl == 0.0
        assert pos.size == 1.0
        assert pos.avg_price == 100.0
        assert log["狀態"] == "開倉"

    def test_close_long_profit(self, mock_config):
        pos = position(mock_config)
        pos.open(price=100.0, size=1.0, tp=110.0, sl=95.0, timestamp='2024-01-01 01:00', entry_index=0)
        
        results = pos.close(price=105.0, size_to_close=-1.0, timestamp='2024-01-01 02:00')
        pnl, log = results[0]
        
        # PnL = (105 - 100) * 1 - 105*0.001 = 5 - 0.105 = 4.895
        assert pnl == pytest.approx(4.895)
        assert pos.size == 0.0
        assert log["狀態"] == "平倉"

    def test_reverse_to_short(self, mock_config):
        pos = position(mock_config)
        pos.open(price=100.0, size=1.0, tp=110.0, sl=95.0, timestamp='2024-01-01 01:00', entry_index=0)
        
        results = pos.reverse(price=95.0, new_size=-1.0, tp=85.5, sl=99.75, timestamp='2024-01-01 03:00', entry_index=1)
        
        pnl_close, log_close = results[0]
        
        # PnL = (95 - 100) * 1 - 95*0.001 = -5 - 0.095 = -5.095
        assert pnl_close == pytest.approx(-5.095)
        assert pos.size == -1.0
        assert pos.avg_price == pytest.approx(95.0)

    def test_pyramiding(self, mock_config):
        mock_config["下單設定"]["pyramiding"] = True
        pos = position(mock_config)

        pos.open(price=100.0, size=1.0, tp=110.0, sl=95.0, timestamp='2024-01-01 01:00', entry_index=0)
        assert pos.size == 1.0
        assert pos.avg_price == 100.0

        pos.open(price=110.0, size=1.0, tp=120.0, sl=100.0, timestamp='2024-01-01 02:00', entry_index=1)
        assert pos.size == 2.0
        assert pos.avg_price == pytest.approx(105.0)

        results = pos.close(price=120.0, size_to_close=-1.0, timestamp='2024-01-01 03:00')
        pnl, log = results[0]
        assert pos.size == 1.0
        # PnL = (120 - 105) * 1 - 120*0.001 = 15 - 0.12 = 14.88
        assert pnl == pytest.approx(14.88)

    def test_slippage_impact(self, mock_config):
        mock_config["回測設定"]["slippage"] = 0.01
        pos = position(mock_config)

        # 測試買入 (做多開倉): 預期成交價 100 * (1 + 0.01) = 101
        results = pos.open(price=100.0, size=1.0, tp=200, sl=90, timestamp='2024-01-01', entry_index=0)
        _, log_open = results[0]
        assert log_open["進場價"] == pytest.approx(101.0)

        # 測試賣出 (平多倉): 預期成交價 200 * (1 - 0.01) = 198
        results = pos.close(price=200.0, size_to_close=-1.0, timestamp='2024-01-02')
        _, log_close = results[0]
        assert log_close["出場價"] == pytest.approx(198.0)

        # 測試做空 (開空倉): 預期成交價 100 * (1 - 0.01) = 99
        pos.size = 0.0 # 重置
        results = pos.open(price=100.0, size=-1.0, tp=90, sl=110, timestamp='2024-01-03', entry_index=0)
        _, log_short = results[0]
        assert log_short["進場價"] == pytest.approx(99.0)

# [Class 說明]
class TestStats:
    """測試 stats class 的計算功能"""
    
    def test_trade_log_and_metrics(self, mock_config):
        stat = stats(mock_config)
        
        profit_log = {'狀態': '平倉', '出場量': -1.0, '實現損益': 100.0}
        stat.trade_log(pnl=100.0, log_event=profit_log)
        
        loss_log = {'狀態': '平倉', '出場量': 1.0, '實現損益': -50.0}
        stat.trade_log(pnl=-50.0, log_event=loss_log)
        
        open_log = {'狀態': '開倉', '進場量': 1.0, '進場價': 100.0}
        stat.trade_log(pnl=0.0, log_event=open_log) 
        
        stat.finalize_log()

        assert stat.count == 2
        assert stat.count_long == 1
        assert stat.count_long_win == 1
        assert stat.count_short == 1
        assert stat.count_short_win == 0
        assert stat.pnl == pytest.approx(49.9)
        assert stat.cash == pytest.approx(10049.9)
        assert stat.winrate() == pytest.approx(50.0)
        assert stat.profit_factor() == pytest.approx(2.0)

    def test_max_drawdown(self, mock_config):
        stat = stats(mock_config) 
        
        stat.trade_log(pnl=100.0, log_event={'狀態': '平倉', '出場量': 1.0, '實現損益': 100.0}) 
        assert stat.max_drawdown == 0.0

        stat.trade_log(pnl=-200.0, log_event={'狀態': '平倉', '出場量': 1.0, '實現損益': -200.0}) 
        assert stat.max_drawdown == pytest.approx(200.0)

        stat.trade_log(pnl=50.0, log_event={'狀態': '平倉', '出場量': 1.0, '實現損益': 50.0}) 
        assert stat.max_drawdown == pytest.approx(200.0)

        stat.trade_log(pnl=-100.0, log_event={'狀態': '平倉', '出場量': 1.0, '實現損益': -100.0}) 
        assert stat.max_drawdown == pytest.approx(250.0)

# [Function 說明]
@patch('backtest.plt.show')
@patch('backtest.result_to_csv')
def test_backtest_run_integration(mock_result_to_csv, mock_plt_show, mock_config, sample_df):
    """對 backtest.run() 進行整合測試，驗證完整流程"""
    bt = backtest(sample_df, mock_config)
    bt.run()
    
    # 預期流程 (已計入開倉和平倉手續費):
    # 1. i=0, open long@100. open_fee=0.1.
    # 2. i=1, tp hit@110. close long. gross_pnl=10. close_fee=0.11. Net Pnl = 10 - 0.1 - 0.11 = 9.79
    # 3. i=2, open short@95. open_fee=0.095.
    # 4. i=3, sl hit@99.75. close short. gross_pnl=-4.75. close_fee=0.09975. Net Pnl = -4.75 - 0.095 - 0.09975 = -4.94475
    # Total Pnl = 9.79 - 4.94475 = 4.84525
    # Final cash = 10000 + 4.84525 = 10004.84525

    assert bt.stats.count == 2
    assert bt.stats.pnl == pytest.approx(4.84525)
    assert bt.stats.cash == pytest.approx(10004.84525)
    mock_result_to_csv.assert_called_once()

def test_insufficient_funds(mock_config, sample_df, caplog):
    """測試當資金不足時，是否會正確地阻止開倉"""
    mock_config["回測設定"]["initial_cash"] = 50.0 
    
    bt = backtest(sample_df, mock_config)
    
    with caplog.at_level(logging.WARNING):
        bt.run()

    assert bt.stats.count == 0
    assert bt.stats.cash == pytest.approx(50.0)
    assert any("[INSUFFICIENT_FUNDS]" in record.message for record in caplog.records)

def test_force_close_by_max_hold(mock_config):
    data = {
        'close_time': pd.to_datetime(['2024-01-01 01:00', '2024-01-01 02:00', '2024-01-01 03:00', '2024-01-01 04:00']),
        'close': [100.0, 105.0, 110.0, 115.0],
        'signal': [1, 0, 0, 0] 
    }
    df = pd.DataFrame(data)

    mock_config["基本設定"]["max_hold"] = 2
    bt = backtest(df, mock_config)
    bt.run()

    assert bt.stats.count == 1
    assert bt.position.size == 0.0
    # Net PnL = (110-100)*1 - (100*0.001) - (110*0.001) = 10 - 0.1 - 0.11 = 9.79
    assert bt.stats.pnl == pytest.approx(9.79)

@patch('backtest.position.open', MagicMock(return_value=[(0.0, {})]))
def test_leverage_trading(mock_config):
    mock_config["下單設定"]["leverage"] = 3
    mock_config["下單設定"]["order_value"] = 1.0
    
    real_pos = position(mock_config)
    bt = backtest(pd.DataFrame(), mock_config)
    bt.position = real_pos

    with patch.object(real_pos, 'open') as mock_open:
        bt._create_order(close=100.0, direction=1, timestamp='2024-01-01', i=0)
        mock_open.assert_called_once()
        called_args, _ = mock_open.call_args
        assert called_args[1] == pytest.approx(3.0)

def test_order_mode_percent(mock_config):
    mock_config["下單設定"]["order_mode"] = "percent"
    mock_config["下單設定"]["order_value"] = 10.0 
    
    bt = backtest(pd.DataFrame(), mock_config) 
    bt._create_order(close=100.0, direction=1, timestamp='2024-01-01', i=0)
    assert bt.position.size == pytest.approx(10.0)

# [Class 說明]
class TestEdgeCases:
    """專門測試邊界條件和例外情況"""

    def test_empty_dataframe(self, mock_config):
        bt = backtest(pd.DataFrame(), mock_config)
        bt.run() # Should return early and not crash
        assert bt.stats.count == 0
        assert bt.stats.pnl == 0.0

    def test_close_non_existent_position(self, mock_config):
        pos = position(mock_config)
        assert pos.size == 0.0
        with pytest.raises(Exception, match="當前無持倉，無法平倉"):
            pos.close(price=100.0, size_to_close=-1.0, timestamp='2024-01-01')

    def test_zero_order_value(self, mock_config):
        pos = position(mock_config)
        with pytest.raises(ValueError, match="size不能為0"):
             pos.open(100.0, 0.0, 110.0, 90.0, 'ts', 0)

    def test_division_by_zero_on_price(self, mock_config):
        mock_config["下單設定"]["order_mode"] = "percent"
        bt = backtest(pd.DataFrame(), mock_config)
        
        with pytest.raises(ZeroDivisionError):
            bt._create_order(close=0.0, direction=1, timestamp='2024-01-01', i=0)

    def test_no_loss_profit_factor(self, mock_config):
        stat = stats(mock_config)
        profit_log = {'狀態': '平倉', '出場量': 1.0, '實現損益': 100.0}
        stat.trade_log(pnl=100.0, log_event=profit_log)
        stat.finalize_log()
        assert stat.profit_factor() == np.inf

    def test_consecutive_reverse_signals(self, mock_config):
        """測試連續的反手信號是否能被正確處理"""
        data = {
            'close_time': pd.to_datetime(['2024-01-01 01:00', '2024-01-01 02:00']),
            'close': [100.0, 105.0],
            'signal': [1, -1]
        }
        df = pd.DataFrame(data)
        bt = backtest(df, mock_config)
        bt.run()

        # 預期 (計入所有手續費):
        # 1. i=0, open long@100. pnl = -0.1 (open_fee)
        # 2. i=1, reverse@105. 
        #    - close long: pnl = -0.1 + (5 - 0.105) = 4.795
        #    - open short: pnl = 4.795 - 0.105 = 4.69
        # 3. end, close short@105.
        #    - pnl = 4.69 + (0 - 0.105) = 4.585
        # Final Cash = 10000 + 4.585 = 10004.585
        assert bt.stats.count == 2 
        assert bt.position.size == 0.0
        assert bt.stats.pnl == pytest.approx(4.585)
        assert bt.stats.cash == pytest.approx(10004.585)