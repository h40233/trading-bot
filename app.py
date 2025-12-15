import streamlit as st
import json
import pandas as pd
import os
import glob
import plotly.express as px
import itertools
import inspect
from decimal import Decimal

# 引入你的專案模組
from util import load_config, to_timestamp, data_to_csv, load_strategy
from backtest import backtest
from data_loader import fetch_and_process_data 
# 引入策略註冊表
from strategies import STRATEGY_REGISTRY

# --- 頁面基礎設定 ---
st.set_page_config(page_title="量化交易控制台", layout="wide", page_icon="📈")
st.title("📈 程式夥伴 - 量化交易控制台")

# --- 狀態管理 (Session State) ---
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# --- Helper: 讀取 config ---
def get_current_config():
    try:
        return load_config()
    except:
        return {}

# --- 側邊欄：系統設定 ---
st.sidebar.header("⚙️ 參數設定")
try:
    config = load_config()
except:
    st.sidebar.error("找不到 config.json")
    config = {}

with st.sidebar.form("config_form"):
    # 1. 基本設定
    with st.expander("📝 基本環境設定 (Basic)", expanded=False):
        col_b1, col_b2 = st.columns(2)
        base_conf = config.get("基本設定", {})
        symbol = col_b1.text_input("交易對", base_conf.get("symbol", "BTCUSDT"))
        timeframe = col_b2.selectbox("時間週期", ["1m", "5m", "15m", "1h", "4h", "1d"], 
                                     index=["1m", "5m", "15m", "1h", "4h", "1d"].index(base_conf.get("timeframe", "5m")))
        
        # 動態讀取所有已註冊的策略
        strategy_options = list(STRATEGY_REGISTRY.keys())
        current_strat = base_conf.get("strategy", "EMA_RSI")
        strat_index = strategy_options.index(current_strat) if current_strat in strategy_options else 0
        strategy_name = st.selectbox("選擇策略", strategy_options, index=strat_index)
        
        # [修正] 補回 checkbox，這樣介面上才會出現勾選框
        col_b3, col_b4 = st.columns(2)
        testnet = col_b3.checkbox("使用測試網 (Testnet)", value=base_conf.get("testnet", True))
        use_mark = col_b4.checkbox("使用標記價格 K線", value=base_conf.get("use_mark_price_kline", False))
        
        col_b5, col_b6 = st.columns(2)
        max_hold = col_b5.number_input("最大持倉 K 棒數 (0為不限)", value=int(base_conf.get("max_hold", 10)))
        fetch_limit = col_b6.number_input("單次抓取 K 線數量", value=int(base_conf.get("fetch_limit", 1000)))
        
        sleep_time = st.number_input("API 冷卻秒數", value=float(base_conf.get("sleep_time", 0.5)))

    # 2. 下單設定
    with st.expander("💰 下單資金管理 (Order)", expanded=False):
        order_conf = config.get("下單設定", {})
        col_o1, col_o2 = st.columns(2)
        order_mode = col_o1.selectbox("下單模式", ["percent", "fixed", "price"], index=["percent", "fixed", "price"].index(order_conf.get("order_mode", "percent")))
        order_value = col_o2.number_input("下單數值", value=float(order_conf.get("order_value", 10)))
        leverage = st.number_input("槓桿倍數", value=int(order_conf.get("leverage", 1)))
        
        # [修正] 把這兩行補回來，介面才會顯示勾選框
        col_o3, col_o4 = st.columns(2)
        pyramiding = col_o3.checkbox("允許加倉 (Pyramiding)", value=order_conf.get("pyramiding", False))
        reverse = col_o4.checkbox("允許反手 (Reverse)", value=order_conf.get("reverse", False))

    # 3. 止盈止損設定
    with st.expander("🛡️ 止盈止損設定 (TP/SL)", expanded=False):
        tpsl_conf = config.get("止盈止損設定", {})
        col_t1, col_t2 = st.columns(2)
        tp_percent = col_t1.checkbox("TP %模式", value=tpsl_conf.get("tp_of_percent", True))
        tp_value = col_t2.number_input("TP 數值", value=float(tpsl_conf.get("tp_value", 0.0)))
        col_s1, col_s2 = st.columns(2)
        sl_percent = col_s1.checkbox("SL %模式", value=tpsl_conf.get("sl_of_percent", True))
        sl_value = col_s2.number_input("SL 數值", value=float(tpsl_conf.get("sl_value", 0.0)))

    # 4. 回測設定
    with st.expander("⏳ 回測環境設定 (Backtest)", expanded=False):
        bt_conf = config.get("回測設定", {})
        start_time_str = st.text_input("開始時間", bt_conf.get("start_time", "2023-01-01 00:00"))
        end_time_str = st.text_input("結束時間", bt_conf.get("end_time", "2023-12-31 00:00"))
        initial_cash = st.number_input("初始資金", value=float(bt_conf.get("initial_cash", 10000)))
        fee_rate = st.number_input("手續費率", value=float(bt_conf.get("fee_rate", 0.0004)), format="%.5f")
        slippage = st.number_input("滑價率", value=float(bt_conf.get("slippage", 0.0005)), format="%.5f")

    submitted = st.form_submit_button("💾 儲存並更新設定")
    
    if submitted:
        config["基本設定"].update({"symbol": symbol, "timeframe": timeframe, "strategy": strategy_name, "max_hold": max_hold if max_hold > 0 else None})
        config["下單設定"].update({"order_mode": order_mode, "order_value": order_value, "leverage": leverage})
        config["止盈止損設定"].update({"tp_of_percent": tp_percent, "tp_value": tp_value, "sl_of_percent": sl_percent, "sl_value": sl_value})
        config["回測設定"].update({"start_time": start_time_str, "end_time": end_time_str, "initial_cash": initial_cash, "fee_rate": fee_rate, "slippage": slippage})
        
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        st.success("設定已儲存！")
        st.cache_data.clear()

# --- 主分頁 ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 單次回測", "🧪 參數優化實驗室", "🤖 實盤監控", "📂 檔案管理"])

# ==========================================
# 分頁 1: 單次回測 (Backtest)
# ==========================================
with tab1:
    st.subheader("歷史回測模擬")
    st.info(f"當前目標：{symbol} | 策略：{strategy_name} | 週期：{timeframe}")
    
    if st.button("🚀 開始回測", type="primary", use_container_width=True):
        status_box = st.empty()
        bar = st.progress(0)
        try:
            status_box.text("正在更新資料與計算策略...")
            bar.progress(10)
            df_signal = fetch_and_process_data(custom_config=config)
            
            if df_signal is not None:
                status_box.text("正在執行回測模擬...")
                bt = backtest(df_signal, config)
                
                def update_progress(p):
                    bar.progress(p, text=f"回測進度: {int(p*100)}%")
                bt.run(progress_callback=update_progress)
                
                bar.progress(1.0, text="回測完成！")
                status_box.success("回測完成！")
                
                st.divider()
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("💰 最終權益", f"{bt.stats.cash:,.2f} U", delta=f"{bt.stats.pnl:,.2f} U")
                col2.metric("🎯 勝率", f"{bt.stats.winrate():.2f} %")
                col3.metric("📉 最大回撤", f"{bt.stats.max_drawdown:,.2f} U")
                col4.metric("📊 夏普比率", f"{bt.stats.sharpe():.2f}")
                
                st.subheader("資金曲線")
                equity = bt.stats.get_equity_curve()
                if equity is not None:
                    st.line_chart(equity, x="時間", y="資金曲線", color="#00FF00")
                
                with st.expander("查看詳細交易日誌"):
                    st.dataframe(bt.stats.log, use_container_width=True)
            else:
                status_box.error("資料獲取失敗")
        except Exception as e:
            st.error(f"錯誤: {e}")

# ==========================================
# 分頁 2: 參數優化實驗室 (Optimization)
# ==========================================
with tab2:
    st.header("🧪 參數優化與穩健性分析")
    st.markdown("此功能自動偵測策略參數，並使用網格搜索 (Grid Search) 尋找參數高原。")
    
    # --- 動態參數偵測邏輯 ---
    if strategy_name in STRATEGY_REGISTRY:
        StrategyClass = STRATEGY_REGISTRY[strategy_name]
        
        # 使用 inspect 取得 __init__ 參數
        sig = inspect.signature(StrategyClass.__init__)
        all_params = [
            p.name for p in sig.parameters.values() 
            if p.name not in ['self', 'name', 'args', 'kwargs'] 
            and p.default != inspect._empty
            and isinstance(p.default, (int, float))
        ]
        
        with st.container(border=True):
            st.subheader("1. 設定優化範圍")
            
            if len(all_params) < 2:
                st.warning(f"此策略只有 {len(all_params)} 個數值參數，無法進行 2D 熱力圖分析 (至少需要 2 個)。")
                selected_params = []
            else:
                st.info(f"偵測到可優化參數：{all_params}")
                selected_params = st.multiselect("請選擇 2 個參數進行優化 (X軸 與 Y軸)", all_params, max_selections=2)
            
            param_settings = {}
            
            if len(selected_params) == 2:
                col_p1, col_p2, col_split = st.columns(3)
                
                # 參數 1 (X軸)
                p1_name = selected_params[0]
                with col_p1:
                    st.markdown(f"**{p1_name} (X軸)**")
                    default_val = sig.parameters[p1_name].default
                    p1_start = st.number_input(f"{p1_name} 開始", value=float(default_val), key="p1_start")
                    p1_end = st.number_input(f"{p1_name} 結束", value=float(default_val*2), key="p1_end")
                    p1_step = st.number_input(f"{p1_name} 間隔", value=float(5), key="p1_step")
                    param_settings[p1_name] = (p1_start, p1_end, p1_step)

                # 參數 2 (Y軸)
                p2_name = selected_params[1]
                with col_p2:
                    st.markdown(f"**{p2_name} (Y軸)**")
                    default_val_2 = sig.parameters[p2_name].default
                    p2_start = st.number_input(f"{p2_name} 開始", value=float(max(1, default_val_2-5)), key="p2_start")
                    p2_end = st.number_input(f"{p2_name} 結束", value=float(default_val_2+10), key="p2_end")
                    p2_step = st.number_input(f"{p2_name} 間隔", value=float(2), key="p2_step")
                    param_settings[p2_name] = (p2_start, p2_end, p2_step)
                
                with col_split:
                    st.markdown("**資料分割設定**")
                    split_ratio = st.slider("訓練集佔比 (In-Sample %)", 0.1, 0.9, 0.7, 0.05)

        # 執行按鈕
        if st.button("🧪 開始網格搜索", type="primary", use_container_width=True):
            if len(selected_params) != 2:
                st.error("請先選擇兩個參數！")
            else:
                # --- UI 元件準備 (使用 st.empty 佔位) ---
                status_header = st.empty()  # 用來顯示「第幾組 / 總共幾組」
                param_display = st.empty()  # [修改點] 用來顯示「當前參數」，這格會一直被覆蓋，不會變長
                current_bar = st.progress(0) # 單次進度條
                
                status_header.text("正在獲取並清洗原始資料...")
                full_df = fetch_and_process_data(custom_config=config)
                
                if full_df is not None:
                    base_columns = ['open_time', 'open', 'high', 'low', 'close', 'close_time', 'symbol']
                    raw_df = full_df[base_columns].copy()
                    
                    # 生成參數範圍
                    def make_range(start, end, step):
                        vals = []
                        curr = start
                        while curr <= end:
                            vals.append(int(curr) if step % 1 == 0 else curr)
                            curr += step
                        return vals

                    range1 = make_range(*param_settings[selected_params[0]])
                    range2 = make_range(*param_settings[selected_params[1]])
                    
                    param_combinations = list(itertools.product(range1, range2))
                    total_combs = len(param_combinations)
                    
                    results = []
                    split_idx = int(len(raw_df) * split_ratio)
                    
                    # callback
                    def update_realtime_bar(p):
                        current_bar.progress(p, text=f"當前模擬進度: {int(p*100)}%")

                    # 開始迴圈
                    for i, (val1, val2) in enumerate(param_combinations):
                        
                        # 更新進度文字
                        status_header.markdown(f"### 🔄 正在執行第 {i + 1} / {total_combs} 組參數組合")
                        
                        # [修改點] 使用 info 或 text 更新同一個區塊，而不是一直 append
                        param_display.info(f"👉 正在測試參數： **{selected_params[0]}={val1}**, **{selected_params[1]}={val2}**")
                        
                        try:
                            init_params = {}
                            init_params[selected_params[0]] = val1
                            init_params[selected_params[1]] = val2
                            
                            strategy_instance = StrategyClass(**init_params)
                            
                            # 計算訊號
                            temp_df = raw_df.copy()
                            df_with_signal = strategy_instance.generate_signal(temp_df)
                            
                            # 切割
                            df_train = df_with_signal.iloc[:split_idx]
                            df_test = df_with_signal.iloc[split_idx:]
                            
                            # 跑回測
                            bt_train = backtest(df_train, config)
                            bt_train.run(progress_callback=update_realtime_bar)
                            
                            bt_test = backtest(df_test, config)
                            bt_test.run(progress_callback=update_realtime_bar)
                            
                            results.append({
                                selected_params[0]: val1,
                                selected_params[1]: val2,
                                "IS_Sharpe": bt_train.stats.sharpe(),
                                "OS_Sharpe": bt_test.stats.sharpe(),
                            })
                            
                        except Exception as e:
                            print(f"優化失敗: {e}")
                    
                    # 完成後處理
                    current_bar.progress(1.0, text="優化全數完成！")
                    status_header.success(f"✅ 已完成全部 {total_combs} 組參數測試")
                    param_display.empty() # 清掉參數顯示，讓畫面乾淨一點
                    
                    if results:
                        res_df = pd.DataFrame(results)
                        
                        st.divider()
                        v1, v2 = st.columns(2)
                        
                        x_axis = selected_params[0]
                        y_axis = selected_params[1]
                        
                        with v1:
                            st.subheader("🔥 訓練集 (In-Sample)")
                            fig1 = px.density_heatmap(
                                res_df, x=x_axis, y=y_axis, z="IS_Sharpe", 
                                text_auto=".2f", color_continuous_scale="RdBu",
                                title="Sharpe Ratio (Train)"
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with v2:
                            st.subheader("❄️ 測試集 (Out-Sample)")
                            fig2 = px.density_heatmap(
                                res_df, x=x_axis, y=y_axis, z="OS_Sharpe", 
                                text_auto=".2f", color_continuous_scale="RdBu",
                                title="Sharpe Ratio (Test)"
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        st.subheader("詳細數據")
                        st.dataframe(res_df, use_container_width=True)
    else:
        st.error(f"找不到策略 {strategy_name}")

# ==========================================
# 分頁 3: 實盤監控
# ==========================================
with tab3:
    st.subheader("實盤運行控制台")
    if st.session_state.is_running:
        st.success("🟢 策略執行中 (Running)")
    else:
        st.warning("🔴 策略已停止 (Stopped)")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶️ 啟動實盤策略", use_container_width=True):
            st.session_state.is_running = True
            st.rerun()
    with col_btn2:
        if st.button("🛑 緊急停止 / Stop All", type="primary", use_container_width=True):
            st.session_state.is_running = False
            st.rerun()

    st.write("---")
    st.markdown("### 📋 即時交易日誌")
    log_files = glob.glob("result/logs/*.csv")
    if log_files:
        latest_file = max(log_files, key=os.path.getctime)
        try:
            live_df = pd.read_csv(latest_file)
            st.dataframe(live_df.tail(15).sort_index(ascending=False), use_container_width=True)
        except:
            st.write("讀取日誌失敗")
    else:
        st.info("尚無交易紀錄")

# ==========================================
# 分頁 4: 檔案管理
# ==========================================
with tab4:
    st.subheader("本地資料管理")
    c1, c2, c3 = st.columns(3)
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