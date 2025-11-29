https://github.com/h40233/trading-bot.git
```
專案
|-- main.py 實盤執行的主程式
|-- backtest.py 用來回測的程式
|-- config.json 用來存放策略需要的參數
|-- .env 用來存放API KEY、SECRET、TG ID等帶隱私的參數
|-- .env.sample 
|-- util.py 用來存放各種工具函式
|-- README.md
|-- requirement.txt
|-- .gitignore
|-- strategies/ 用來存放各策略的資料夾
|       |-- strategy.py 所有策略的父類別
|       |-- ......
|-- models/ 用來存放各模型的資料夾
|-- result/ 用來存放各種結果的資料夾
|       |-- backtests/ 存放回測結果的資料夾
|       |-- log/ 存放交易結果及錯誤訊息的資料夾
|-- data/ 用來存放已經抓過的K線資料
|       |-- raw/ 原始K線CSV
|       |-- processed 處理過帶特徵的CSV
```
量化交易練習
2025.09.13 第一次上傳git
2025.09.16 可以抓取K線資料
2025.09.20 可以動態載入策略，並將帶有signal的df存成csv
2025.11.30 新增一份規格文件，並有對應的單元測試