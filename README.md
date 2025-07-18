# Dueling-DDQN_CV2X_Offloading

## 專案簡介
本專案旨在於5G-CV2X（Cellular Vehicle-to-Everything）車聯網環境下，提出一套基於Dueling Double Deep Q-Network（Dueling-DDQN）的低延遲任務卸載決策機制。透過深度強化學習，動態決定車輛本地端、MEC（多接取邊緣運算）及雲端的資料計算卸載與協作策略，以實現高效能、低延遲與高安全性的車載計算卸載。

## 研究動機
隨著自動駕駛與物聯網技術發展，車載系統產生的數據量大幅增加，對即時資料處理與決策提出更高要求。傳統卸載方法未能兼顧車輛高速移動下的延遲閾值與安全性，因此本研究結合5G-CV2X、MEC與深度強化學習，提出動態、分散式且具備低延遲的卸載決策機制。

## 方法概述
- **Dueling Double DQN**：採用Dueling-DDQN架構，分離狀態價值與動作優勢，提升決策穩定性與收斂速度，並解決Q值高估問題。
- **多層卸載架構**：車輛可根據本地、MEC、雲端的即時狀態，動態選擇卸載路徑。
- **強化學習決策**：以平均任務完成時間最小化為目標，設計狀態、動作、獎勵與價值函數，並於SUMO模擬環境下訓練。

## 系統架構與流程
- 車載系統、MEC、雲端三層架構，車輛可V2V/V2I協作或直接卸載至雲端。
- 以M/G/1排隊模型計算各節點延遲，並以強化學習動態調整卸載決策。
- 詳細流程與架構圖可參考論文及`MC2023徐偉銘`資料夾內相關圖檔。

## 實驗結果摘要
- 本研究於SUMO模擬高速公路與城市道路場景，驗證Dueling-DDQN在不同環境下的效能。
- 實驗顯示，Dueling-DDQN能在車輛高速移動下於延遲閾值內達成99.93%卸載完成率，且收斂速度與穩定性優於DQN、DDQN及全雲端卸載。
- 相關數據與圖表詳見下方展示。

## 如何運行

本專案主要運行流程與實驗重現，請參考 `Code/Dueling_DDQN_offloading/DDQN_main.ipynb` Jupyter Notebook。

### 執行步驟
1. 安裝必要套件：
   - Python 3.7 以上
   - TensorFlow 2.x
   - numpy、pandas、matplotlib、scipy
   - Jupyter Notebook
2. 進入 `Code/Dueling_DDQN_offloading/` 資料夾，於終端機輸入：
   ```bash
   jupyter notebook
   ```
3. 開啟並依序執行 `DDQN_main.ipynb` 內各區塊（Cell），即可重現 Dueling-DDQN 訓練、測試與圖表產生流程。
4. 輸出結果（如訓練分數、延遲、圖表等）將自動儲存於 `plots/`、`data_poisson/` 等資料夾。

# 1. 5G傳輸與DSRC傳輸方式比較
![Image Alt Text](plots/all_avg_score_offloading_plot/offloading_5g_and_DSRC.png)

# 2. 學習率對Dueling-DDQN影響
![Image Alt Text](plots/learning_rate/offloading_all_avg.png)

# 3. 衰減率對Dueling-DDQN影響
![Image Alt Text](plots/GAMMA/offloading_all__gamma_avg.png)

# 4. 不同強化學習在車載環境計算卸載影響 
![Image Alt Text](plots/all_avg_score_offloading_plot/offloading_all_avg.png)

# 5. 不同強化學習計算時間延遲率
![Image Alt Text](plots/localization_latency/localization_latency_CDF.png)

> **圖表說明：**
> 此圖比較了全卸載到雲端、DQN、DDQN以及本研究提出的Dueling-DDQN在相同環境下的訓練分數變化。分數越高代表平均延遲越低。
> - **全雲端卸載**：分數維持在-0.74區間，僅考慮傳輸延遲。
> - **DQN**：在Epoch 350回合後分數穩定，但於Epoch 850後因greedy決策導致分數下降，最終分數為-0.71。
> - **DDQN**：在Epoch 300後分數持續震盪，最終分數為-0.65，避免了DQN的高估問題但穩定性略差。
> - **Dueling-DDQN**：於Epoch 100迅速收斂且分數穩定，最終分數為-0.60，展現出最佳的收斂速度與穩定性，能有效提升卸載決策效能並降低延遲。
> 
> 結論：Dueling-DDQN在車載卸載決策上表現優於其他方法，能更快收斂並維持較低延遲。

