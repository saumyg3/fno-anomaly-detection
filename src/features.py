import pandas as pd
import numpy as np

# =============================================================================
# F&O Anomaly Detection — Feature Engineering Pipeline
# =============================================================================
# Processes 2.5M+ rows of NSE/BSE/MCX F&O data into ML-ready features.
# Run via notebook: notebooks/FnO-Anomaly-Detection.ipynb
# =============================================================================

def load_data(path: str = "data/3mfanddo.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0"])
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="%d-%b-%Y")
    df["EXPIRY_DT"] = pd.to_datetime(df["EXPIRY_DT"], format="%d-%b-%Y")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer ML features from raw F&O data.
    All features are designed to capture anomalous trading behaviour.
    """
    # Work on options only (CE + PE), exclude futures
    options = df[df["OPTION_TYP"].isin(["CE", "PE"])].copy()

    # Sort by symbol and date
    options = options.sort_values(["SYMBOL", "TIMESTAMP"])

    # ── Feature 1: OI Change Rate ──────────────────────────────────────
    # How fast is open interest changing relative to total OI?
    # Sudden spikes = potential anomaly
    options["oi_change_rate"] = options["CHG_IN_OI"] / (options["OPEN_INT"] + 1)

    # ── Feature 2: Volume Z-Score (per symbol) ─────────────────────────
    # How many std deviations away from the mean volume is today's volume?
    options["contracts_zscore"] = options.groupby("SYMBOL")["CONTRACTS"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )

    # ── Feature 3: Rolling 7-day Volatility ───────────────────────────
    # Price range volatility over last 7 trading days
    options["price_range"] = options["HIGH"] - options["LOW"]
    options["rolling_volatility"] = options.groupby("SYMBOL")["price_range"].transform(
        lambda x: x.rolling(7, min_periods=1).std()
    )

    # ── Feature 4: Put-Call Ratio (PCR) ───────────────────────────────
    # High PCR = bearish sentiment, extreme values = potential anomaly
    daily_pcr = options.groupby(["TIMESTAMP", "SYMBOL", "OPTION_TYP"])["OPEN_INT"].sum().unstack(fill_value=0)
    if "PE" in daily_pcr.columns and "CE" in daily_pcr.columns:
        daily_pcr["pcr"] = daily_pcr["PE"] / (daily_pcr["CE"] + 1)
    else:
        daily_pcr["pcr"] = 0
    daily_pcr = daily_pcr[["pcr"]].reset_index()
    options = options.merge(daily_pcr, on=["TIMESTAMP", "SYMBOL"], how="left")

    # ── Feature 5: Expiry Day Flag ─────────────────────────────────────
    # Trading behaviour changes dramatically near expiry
    options["days_to_expiry"] = (options["EXPIRY_DT"] - options["TIMESTAMP"]).dt.days
    options["is_expiry_week"] = (options["days_to_expiry"] <= 7).astype(int)

    # ── Feature 6: Value Concentration ────────────────────────────────
    # Is a disproportionate amount of value concentrated in this contract?
    options["value_per_contract"] = options["VAL_INLAKH"] / (options["CONTRACTS"] + 1)

    # ── Feature 7: OI Surge ────────────────────────────────────────────
    # Absolute OI change vs rolling mean — flags unusual accumulation
    options["oi_surge"] = options.groupby("SYMBOL")["CHG_IN_OI"].transform(
        lambda x: x / (x.rolling(7, min_periods=1).mean().abs() + 1)
    )

    # Drop rows with NaN features
    feature_cols = [
        "oi_change_rate", "contracts_zscore", "rolling_volatility",
        "pcr", "is_expiry_week", "value_per_contract", "oi_surge"
    ]
    options = options.dropna(subset=feature_cols)

    return options, feature_cols


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df):,} rows")

    print("Engineering features...")
    featured, feature_cols = engineer_features(df)
    print(f"After feature engineering: {len(featured):,} rows")
    print(f"Features: {feature_cols}")
    print(featured[feature_cols].describe())

    # Save for use by ML models
    featured.to_parquet("data/featured.parquet", index=False)
    print("Saved to data/featured.parquet")