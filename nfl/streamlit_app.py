import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="NFL Model Dashboard", layout="wide")

st.title("NFL probabilities, edges, and bankroll dashboard")

pred_dir = st.text_input("Predictions directory", "./data/weekly_predictions")
pred_files = [os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.startswith("predictions_") and f.endswith(".csv")] if os.path.exists(pred_dir) else []
if not pred_files:
    st.warning("No prediction CSVs found.  Run the modeling notebook to generate predictions.")
else:
    preds = pd.concat([pd.read_csv(fp) for fp in sorted(pred_files)], ignore_index=True)
    st.subheader("Latest predictions preview")
    st.dataframe(preds.tail(20))

lines_path = st.text_input("Historical lines CSV", "./data/lines_historical.csv")
if os.path.exists(lines_path):
    lines = pd.read_csv(lines_path)
    st.subheader("Historical lines preview")
    st.dataframe(lines.head())
else:
    st.info("Provide a historical lines CSV to enable backtesting.")

st.markdown("---")

# Optional - load backtest results if present
bt_path = "./data/backtest_results.csv"
if os.path.exists(bt_path):
    bt = pd.read_csv(bt_path)
    st.subheader("Backtest results")
    # Bankroll plot
    fig1, ax1 = plt.subplots()
    ax1.plot(bt["bankroll_after"])
    ax1.set_title("Bankroll over time - fractional Kelly")
    ax1.set_xlabel("Bet index")
    ax1.set_ylabel("Bankroll")
    st.pyplot(fig1)

    # Edge histogram
    fig2, ax2 = plt.subplots()
    edges = bt["edge_home"].dropna()
    ax2.hist(edges, bins=30)
    ax2.set_title("Edge distribution - model minus implied")
    ax2.set_xlabel("Edge")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

    # Summary
    total_bets = int((bt["stake"] > 0).sum())
    roi = (bt["pnl"].sum() / bt["stake"].sum()) if bt["stake"].sum() > 0 else np.nan
    hit_rate = float((bt.loc[bt["stake"] > 0, "home_win"] == 1).mean()) if total_bets > 0 else np.nan
    st.write({
        "total_bets": total_bets,
        "roi": roi,
        "hit_rate": hit_rate,
        "final_bankroll": float(bt["bankroll_after"].iloc[-1]) if len(bt) else np.nan
    })
else:
    st.info("No backtest_results.csv found.  Run the backtest in the notebook to generate it.")
