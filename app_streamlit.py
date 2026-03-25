"""
Streamlit dashboard for quantization benchmark results.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from quant_pipeline.core.pipeline import Pipeline

st.title("Quantization Dashboard")

st.header("Single Prediction")
text = st.text_input("Input text", placeholder="Enter a sentence...")
mode = st.selectbox("Precision Mode", ["fp32", "fp16", "int8_ptq", "int8_qat"])

if st.button("Predict"):
    pipe = Pipeline(precision=mode)
    result = pipe.predict(text)
    label_map = {0: "NEGATIVE", 1: "POSITIVE"}
    st.write(f"**Prediction:** {label_map.get(result['label'], result['label'])}")
    st.write(f"**Confidence:** {result['confidence']:.3f}")
    st.write(f"**Mode:** {result['mode']}")

st.header("Benchmark Results")
try:
    df = pd.read_csv("outputs/results.csv")
    st.dataframe(df)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(df["mode"], df["accuracy"], color="#1976d2")
    axes[0].set_title("Accuracy")
    axes[0].set_ylim(0, 1.05)

    axes[1].bar(df["mode"], df["avg_latency_ms"], color="#388e3c")
    axes[1].set_title("Avg Latency (ms)")

    axes[2].bar(df["mode"], df["memory_mb"], color="#d32f2f")
    axes[2].set_title("Memory (MB)")

    plt.tight_layout()
    st.pyplot(fig)
except FileNotFoundError:
    st.warning("Run `python scripts/run_benchmark.py` first to generate results.")

st.header("Robustness Under Perturbations")
try:
    rob_df = pd.read_csv("outputs/robustness.csv")
    st.dataframe(rob_df)
except FileNotFoundError:
    st.info("Robustness data will appear after running the benchmark.")