import streamlit as st
import json
import pandas as pd
import os
import glob
from decimal import Decimal

# 引入你的專案模組
# 注意：這裡引入了我們剛重構好的 data_loader
from util import load_config, to_timestamp, data_to_csv, load_strategy
from backtest import backtest
from data_loader import fetch_and_process_data 

# --- 頁面基礎設定 ---
st.set_page_config(page_title="量化交易控制台", layout="wide", page_icon="📈")
st.title("📈 程式夥伴 - 量化交易控制台")

# --- 狀態管理 (Session State) ---
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# --- 側邊欄：系統設定 (對應 config.json) ---
st.sidebar.header("⚙️ 參數設定")

# 讀取設定檔 (不快取，確保每次存檔後重整都能讀到最新)
try:
    config = load_config()
except:
    st.sidebar.error("找不到 config.json，請檢查檔案位置")
    config = {}

# 使用 Form 表單，避免每次修改一個數字就重新整理頁面
with st.sidebar.form("config_form"):
    
    # 1. 基本設定
    with st.expander("📝 基本環境設定 (Basic)", expanded=True):
        col_b1, col_b2 = st.columns(2)
        base_conf = config.get("基本設定", {})
        
        symbol = col_b1.text_input("交易對 (Symbol)", base_conf.get("symbol", "BTCUSDT"))
        timeframe = col_b2.selectbox("時間週期", ["1m", "5m", "15m", "1h", "4h", "1d"], 
                                     index=["1m", "5m", "15m", "1h", "4h", "1d"].index(base_conf.get("timeframe", "5m")))
        
        strategy_name = st.text_input("策略名稱", base_conf.get("strategy", "EMA_RSI"))
        
        col_b3, col_b4 = st.columns(2)
        testnet = col_b3.checkbox("使用測試網 (Testnet)", value=base_conf.get("testnet", True))
        use_mark = col_b4.checkbox("使用標記價格 K線", value=base_conf.get("use_mark_price_kline", False))
        
        col_b5, col_b6 = st.columns(2)
        max_hold = col_b5.number_input("最大持倉 K 棒數 (0為不限)", value=int(base_conf.get("max_hold", 10)))
        fetch_limit = col_b6.number_input("單次抓取 K 線數量", value=int(base_conf.get("fetch_limit", 1000)))
        
        col_b7, col_b8, col_b9 = st.columns(3)
        sleep_time = col_b7.number_input("API 冷卻秒數", value=float(base_conf.get("sleep_time", 0.5)))
        retry_wait = col_b8.number_input("重試等待秒數", value=int(base_conf.get("retry_wait", 10)))
        retry_count = col_b9.number_input("重試次數", value=int(base_conf.get("retry_count", 3)))

    # 2. 下單設定
    with st.expander("💰 下單資金管理 (Order)", expanded=False):
        order_conf = config.get("下單設定", {})
        
        col_o1, col_o2 = st.columns(2)
        order_mode = col_o1.selectbox("下單模式", ["percent", "fixed", "price"], 
                                      index=["percent", "fixed", "price"].index(order_conf.get("order_mode", "percent")))
        order_value = col_o2.number_input("下單數值 (佔比/數量/金額)", value=float(order_conf.get("order_value", 10)))
        
        leverage = st.number_input("槓桿倍數", value=int(order_conf.get("leverage", 1)), min_value=1, max_value=125)
        
        col_o3, col_o4 = st.columns(2)
        pyramiding = col_o3.checkbox("允許加倉 (Pyramiding)", value=order_conf.get("pyramiding", False))
        reverse = col_o4.checkbox("允許反手 (Reverse)", value=order_conf.get("reverse", False))

    # 3. 止盈止損設定
    with st.expander("🛡️ 止盈止損設定 (TP/SL)", expanded=False):
        tpsl_conf = config.get("止盈止損設定", {})
        
        st.caption("止盈 (Take Profit)")
        col_t1, col_t2 = st.columns(2)
        tp_percent = col_t1.checkbox("TP 使用百分比模式", value=tpsl_conf.get("tp_of_percent", True))
        tp_value = col_t2.number_input("TP 數值 (百分比/價差)", value=float(tpsl_conf.get("tp_value", 0.0)))
        
        st.caption("止損 (Stop Loss)")
        col_s1, col_s2 = st.columns(2)
        sl_percent = col_s1.checkbox("SL 使用百分比模式", value=tpsl_conf.get("sl_of_percent", True))
        sl_value = col_s2.number_input("SL 數值 (百分比/價差)", value=float(tpsl_conf.get("sl_value", 0.0)))

    # 4. 回測設定
    with st.expander("⏳ 回測環境設定 (Backtest)", expanded=False):
        bt_conf = config.get("回測設定", {})
        
        start_time_str = st.text_input("開始時間", bt_conf.get("start_time", "2023-01-01 00:00"))
        end_time_str = st.text_input("結束時間", bt_conf.get("end_time", "2023-12-31 00:00"))
        
        col_bt1, col_bt2, col_bt3 = st.columns(3)
        initial_cash = col_bt1.number_input("初始資金 (U)", value=float(bt_conf.get("initial_cash", 10000)))
        fee_rate = col_bt2.number_input("手續費率", value=float(bt_conf.get("fee_rate", 0.0004)), format="%.5f")
        slippage = col_bt3.number_input("滑價率 (Slippage)", value=float(bt_conf.get("slippage", 0.0005)), format="%.5f")

    # 送出按鈕
    submitted = st.form_submit_button("💾 儲存並更新設定")
    
    if submitted:
        # 更新 Config 字典
        config["基本設定"].update({
            "symbol": symbol, "timeframe": timeframe, "strategy": strategy_name,
            "testnet": testnet, "use_mark_price_kline": use_mark,
            "max_hold": max_hold if max_hold > 0 else None, # 如果是 0 改為 None
            "fetch_limit": fetch_limit, "sleep_time": sleep_time,
            "retry_wait": retry_wait, "retry_count": retry_count
        })
        
        config["下單設定"].update({
            "order_mode": order_mode, "order_value": order_value,
            "leverage": leverage, "pyramiding": pyramiding, "reverse": reverse
        })
        
        config["止盈止損設定"].update({
            "tp_of_percent": tp_percent, "tp_value": tp_value,
            "sl_of_percent": sl_percent, "sl_value": sl_value
        })
        
        config["回測設定"].update({
            "start_time": start_time_str, "end_time": end_time_str,
            "initial_cash": initial_cash, "fee_rate": fee_rate, "slippage": slippage
        })
        
        # 寫入檔案
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        st.success("設定已儲存！")
        st.cache_data.clear() # 清除讀取快取

# --- 主分頁 ---
tab1, tab2, tab3 = st.tabs(["📊 回測系統", "🤖 實盤監控", "📂 檔案管理"])

# ==========================================
# 分頁 1: 回測系統 (Backtest)
# ==========================================
with tab1:
    st.subheader("歷史回測模擬")
    st.info(f"當前目標：{symbol} | 策略：{strategy_name} | 週期：{timeframe} | 槓桿：{leverage}x")
    
    col_act1, col_act2 = st.columns([1, 4])
    with col_act1:
        start_btn = st.button("🚀 開始回測", type="primary", use_container_width=True)
    
    if start_btn:
        status_box = st.empty()
        bar = st.progress(0)
        
        try:
            # 1. 透過 data_loader 獲取並計算資料
            # 我們直接把上面表單更新過的 config 傳進去，這樣不用重讀檔案
            status_box.text("正在更新資料與計算策略...")
            bar.progress(30)
            
            # 呼叫 data_loader 的核心函式
            df_signal = fetch_and_process_data(custom_config=config)
            
            if df_signal is not None:
                # 這裡原本是 bar.progress(60)，我們可以改成一個文字提示，因為接下來 bar 會交給回測控制
                status_box.text("正在執行回測模擬...")
                
                # 2. 執行回測
                bt = backtest(df_signal, config)
                
                # --- 修改重點 ---
                # 定義一個 callback 函式，用來更新 streamlit 的 bar
                def update_progress(p):
                    # p 是 0.0 到 1.0 的浮點數
                    bar.progress(p, text=f"回測進度: {int(p*100)}%")
                
                # 把這個函式傳進去
                bt.run(progress_callback=update_progress)
                # ---------------
                
                # 跑完後確保滿格
                bar.progress(1.0, text="回測完成！")
                status_box.success("回測完成！")
                
                # 3. 顯示結果指標
                st.divider()
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("💰 最終權益", f"{bt.stats.cash:,.2f} U", delta=f"{bt.stats.pnl:,.2f} U")
                col2.metric("🎯 勝率", f"{bt.stats.winrate():.2f} %", help=f"多單: {bt.stats.long_winrate():.1f}% / 空單: {bt.stats.short_winrate():.1f}%")
                col3.metric("📉 最大回撤", f"{bt.stats.max_drawdown:,.2f} U", help=f"卡瑪比率: {bt.stats.calmar_ratio():.2f}")
                col4.metric("📊 夏普比率", f"{bt.stats.sharpe():.2f}", help=f"獲利因子: {bt.stats.profit_factor():.2f}")
                
                # 4. 繪圖
                st.subheader("資金曲線 (Equity Curve)")
                equity = bt.stats.get_equity_curve()
                if equity is not None:
                    st.line_chart(equity, x="時間", y="資金曲線", color="#00FF00")
                else:
                    st.warning("交易次數不足，無法繪製圖表")

                # 5. 詳細日誌
                with st.expander("查看詳細交易日誌 (Trade Log)"):
                    st.dataframe(bt.stats.log, use_container_width=True)
            else:
                status_box.error("資料獲取失敗，請檢查終端機 Log 或 API 設定")
                
        except Exception as e:
            st.error(f"執行過程中發生錯誤: {e}")

# ==========================================
# 分頁 2: 實盤監控 (Live)
# ==========================================
with tab2:
    st.subheader("實盤運行控制台")
    
    if st.session_state.is_running:
        st.success("🟢 策略執行中 (Running)")
    else:
        st.warning("🔴 策略已停止 (Stopped)")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶️ 啟動實盤策略", use_container_width=True):
            st.session_state.is_running = True
            st.toast("策略已啟動！")
            st.rerun()

    with col_btn2:
        if st.button("🛑 緊急停止 / Stop All", type="primary", use_container_width=True):
            st.session_state.is_running = False
            st.error("已發送緊急停止信號！")
            st.rerun()

    st.write("---")
    st.markdown("### 📋 即時交易日誌 (Live Logs)")
    
    # 讀取 logs 資料夾下最新的 csv
    log_files = glob.glob("result/logs/*.csv")
    if log_files:
        latest_file = max(log_files, key=os.path.getctime)
        st.caption(f"監控日誌來源: {latest_file}")
        try:
            live_df = pd.read_csv(latest_file)
            st.dataframe(live_df.tail(15).sort_index(ascending=False), use_container_width=True)
        except:
            st.write("讀取日誌失敗")
    else:
        st.info("尚無交易紀錄")

# ==========================================
# 分頁 3: 檔案管理 (Files)
# ==========================================
with tab3:
    st.subheader("本地資料管理")
    
    c1, c2, c3 = st.columns(3)
    
    # 掃描檔案
    raw_files = glob.glob("data/raw/*.csv")
    processed_files = glob.glob("data/processed/*.csv")
    result_files = glob.glob("result/backtests/*.csv")
    
    with c1:
        st.write(f"📁 原始 K 線 ({len(raw_files)})")
        if raw_files:
            st.dataframe(pd.DataFrame([os.path.basename(f) for f in raw_files], columns=["檔名"]), hide_index=True)
            
    with c2:
        st.write(f"📁 策略訊號 ({len(processed_files)})")
        if processed_files:
            st.dataframe(pd.DataFrame([os.path.basename(f) for f in processed_files], columns=["檔名"]), hide_index=True)

    with c3:
        st.write(f"📁 回測結果 ({len(result_files)})")
        if result_files:
            st.dataframe(pd.DataFrame([os.path.basename(f) for f in result_files], columns=["檔名"]), hide_index=True)