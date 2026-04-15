# 📊 F&O Anomaly Detection

> 3-model ensemble anomaly detection system trained on 2.5M+ real NSE/BSE/MCX Futures & Options trading records. Detects unusual option activity using engineered financial features and compares Isolation Forest, LSTM Autoencoder, and DBSCAN approaches.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-ee4c2c?style=flat-square&logo=pytorch)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🧠 What Is This?

This project builds a production-grade anomaly detection pipeline on real Indian derivatives market data. Instead of using synthetic or toy datasets, every model is trained on **2,533,210 rows** of actual NSE/BSE/MCX F&O trade records from August–October 2019.

The system flags unusual trading activity that could indicate:
- **Market manipulation** — abnormal OI accumulation
- **Informed trading** — unusual PCR shifts before events
- **Systemic risk signals** — correlated volatility spikes across contracts
- **Expiry-driven anomalies** — irregular activity in expiry weeks

---

## 🌏 Market Context

This project analyzes the **Indian derivatives market** — one of the largest and most liquid options markets in the world. NSE (National Stock Exchange) consistently ranks among the **top 3 global exchanges by options volume**.

Key instruments in this dataset:
- **NIFTY** — India's benchmark index (equivalent to S&P 500)
- **BANKNIFTY** — Banking sector index, the most volatile and actively traded options contract in India
- **Individual stocks** — RELIANCE, INFY, HDFC, SBIN and 300+ others

**Why this matters for US-based roles:** The anomaly detection techniques, feature engineering patterns, and ML architecture used here are **directly transferable** to US equity and options markets (SPX, VIX, individual equities). The underlying market microstructure — OI dynamics, PCR signals, expiry effects — behaves similarly across global derivatives markets.

The dataset covers **August–October 2019**, a period that included the Saudi Aramco oil attack (Sept 14, 2019) — a real geopolitical shock that caused measurable market stress, which our unsupervised models detected without any labeled data.

## 🔍 Key Results

| Model | Anomalies Detected | Rate |
|---|---|---|
| Isolation Forest | 125,054 | 5.0% |
| LSTM Autoencoder | 241,421 | 9.7% |
| DBSCAN | 842 (subsample) | 1.7% |
| **Ensemble** | **78,404** | **3.1%** |
| All 3 models agree | 651 | High confidence |

### 🗓️ Real-World Validation
The model flagged **September 20, 2019** as the most anomalous NIFTY trading day — 4 days after the **Saudi Aramco oil attack** (Sept 14, 2019) caused global market volatility. The system detected the downstream market stress without any labeled training data.

### 📈 Top Anomalous Symbols
BANKNIFTY showed the highest anomaly rate (19%), consistent with its position as India's most volatile options index.

### 🔑 Feature Importance
`value_per_contract` (0.57 correlation) emerged as the strongest anomaly signal — unusual money concentration per trade is more predictive than raw OI or volume alone.

---

## 🏗️ Architecture

```
Raw F&O CSV (2.5M rows)
        │
        ▼
┌─────────────────────────────────────┐
│        Feature Engineering          │
│  7 financial features engineered    │
│  from raw OHLCV + OI data           │
└─────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│                  3-Model Pipeline                     │
│                                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────┐ │
│  │ Isolation Forest│  │ LSTM Autoencoder│  │DBSCAN│ │
│  │  n=200 trees    │  │ 2-layer encoder │  │ eps= │ │
│  │  contamination  │  │ + decoder       │  │ 0.5  │ │
│  │  = 5%           │  │ seq_len=10      │  │      │ │
│  │  Weight: 0.4    │  │ Weight: 0.4     │  │ 0.2  │ │
│  └─────────────────┘  └─────────────────┘  └──────┘ │
│                    │           │                │     │
│                    └─────┬─────┘                │     │
│                          ▼                      │     │
│              ┌───────────────────────┐          │     │
│              │   Weighted Ensemble   │◄─────────┘     │
│              │  score ≥ 0.5 = anomaly│                │
│              └───────────────────────┘                │
└──────────────────────────────────────────────────────┘
        │
        ▼
  Results + Visualizations
```

---

## 🧪 Models Explained

### 1. Isolation Forest
Unsupervised ensemble method that isolates anomalies by randomly partitioning the feature space. Anomalies require fewer splits to isolate — they're "easier to separate" from the crowd. Trained on the full 2.5M row dataset using all CPU cores.

### 2. LSTM Autoencoder
Deep learning approach for time-series anomaly detection. The model learns to reconstruct **normal** trading sequences. When reconstruction error exceeds the 95th percentile threshold, the sequence is flagged as anomalous. Trained on 100k sequences of length 10 using GPU acceleration.

### 3. DBSCAN
Density-based spatial clustering. Trading records that don't belong to any dense cluster are labeled as anomalies (label = -1). Run on a 50k stratified subsample due to O(n²) complexity.

### 4. Weighted Ensemble
Combines all three models with weights (0.4, 0.4, 0.2). A record is flagged as anomalous when the weighted score ≥ 0.5. This reduces false positives from any single model while preserving sensitivity.

---

## ⚙️ Feature Engineering

7 features engineered from raw OHLCV + OI data, each designed to capture a specific dimension of anomalous trading behaviour in the F&O market.

| Feature | Formula | Financial Meaning | Why It Detects Anomalies |
|---|---|---|---|
| `oi_change_rate` | CHG_IN_OI / OPEN_INT | Rate of open interest change relative to total outstanding contracts | Sudden large OI shifts indicate aggressive position buildup — often precedes informed trading or manipulation |
| `contracts_zscore` | (volume - mean) / std per symbol | How many standard deviations today's volume is from that symbol's historical average | Statistically extreme volume (Z > 3) signals unusual market participation |
| `rolling_volatility` | 7-day std of (HIGH - LOW) | Realized volatility of the price range over the last week | Volatility regime shifts — calm markets suddenly becoming erratic — are a key anomaly signal |
| `pcr` | Total Put OI / Total Call OI per day | Put-Call Ratio — a market sentiment indicator | PCR < 0.5 = extreme bullishness, PCR > 1.5 = extreme bearishness. Both extremes are anomalous |
| `is_expiry_week` | 1 if days to expiry ≤ 7, else 0 | Binary flag for the final week before contract expiry | Trading behaviour changes structurally near expiry — rollover activity, pinning, gamma squeezes |
| `value_per_contract` | VAL_INLAKH / CONTRACTS | Average rupee value per contract traded | Unusually high value-per-contract indicates large institutional orders or block trades — a concentration risk signal |
| `oi_surge` | CHG_IN_OI / 7-day rolling mean of CHG_IN_OI | How much today's OI change deviates from the recent baseline | Detects sudden OI accumulation that is anomalous relative to that symbol's own recent history |

---

## 📁 Project Structure

```
fno-anomaly-detection/
├── src/
│   ├── features.py                   # Feature engineering pipeline
│   └── models.py                     # All 3 models + ensemble logic
├── notebooks/
│   └── FnO_Anomaly_Detection.ipynb   # Full end-to-end Colab notebook
├── results/
│   ├── anomaly_detection_results.png # Model comparison charts
│   └── feature_importance.png        # Feature correlation plot
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)
The full pipeline is designed to run on Google Colab with a free T4 GPU. This is the recommended approach since training on 2.5M rows requires GPU acceleration.

1. Open [notebooks/FnO_Anomaly_Detection.ipynb](notebooks/FnO_Anomaly_Detection.ipynb) in Google Colab
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Download the dataset from Kaggle (see below) and upload it to Colab
4. Run all cells in order — the full pipeline takes ~10 minutes on T4 GPU

### Option 2 — Local (Feature Engineering only)
The feature engineering step can run locally. Model training requires GPU for reasonable speed.

```bash
# 1. Clone the repo
git clone https://github.com/saumyg3/fno-anomaly-detection.git
cd fno-anomaly-detection

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset (see below) and place at data/3mfanddo.csv

# 5. Run feature engineering
python src/features.py
# Output: data/featured.parquet

# 6. Train models (GPU recommended)
python src/models.py
# Output: data/results.parquet + saved models
```

---

## 📥 Dataset Setup

1. Go to [NSE Future and Options Dataset 3M](https://www.kaggle.com/datasets/sunnysai12345/nse-future-and-options-dataset-3m) on Kaggle
2. Click **Download dataset as zip (36 MB)**
3. Unzip and place the CSV at:
   - **Local:** `data/3mfanddo.csv`
   - **Colab:** Upload directly when prompted

The dataset contains **2,533,210 rows** of F&O trade records across NSE, BSE, and MCX exchanges from August–October 2019.

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
torch
matplotlib
plotly
seaborn
scipy
```

Install all:
```bash
pip install pandas numpy scikit-learn torch matplotlib plotly seaborn scipy
```

---

## 📊 Visualizations

### Model Comparison + NIFTY Anomaly Timeline
![Model Comparison](results/anomaly_detection_results.png)

### Feature Importance
![Feature Importance](results/feature_importance.png)

---

## 💡 Key Findings

1. **value_per_contract** is the strongest anomaly signal (0.57 correlation) — unusual money per trade is more predictive than raw volume
2. **Expiry weeks** and **rolling volatility** are equally important (0.40 each) — market structure matters as much as raw numbers
3. **BANKNIFTY** has the highest anomaly rate (19%) — consistent with its volatility profile as India's most traded index
4. The **ensemble outperforms** any single model by reducing false positives from LSTM's aggressive flagging while retaining Isolation Forest's precision
5. **Sept 20, 2019** was the most anomalous NIFTY day — correlates directly with post-Aramco global volatility, validating the model without labeled data

---

## 🗺️ Roadmap

- [ ] Streamlit dashboard for interactive anomaly exploration
- [ ] Live NSE data feed integration
- [ ] Transformer-based sequence model (replace LSTM)
- [ ] Alert system for high-confidence anomalies
- [ ] Backtesting: do flagged anomalies predict next-day price moves?

---

## 👤 Author

**Saumya Goyal**
- GitHub: [@saumyg3](https://github.com/saumyg3)
- LinkedIn: [linkedin.com/in/saumyagoyal](https://linkedin.com/in/saumyagoyal)
- Email: saumyg3@uci.edu

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Dataset: NSE/BSE/MCX F&O data (Aug–Oct 2019) via Kaggle. All models trained without labeled anomaly ground truth — fully unsupervised.*
