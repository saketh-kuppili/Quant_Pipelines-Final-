# Layer-Aware Quantization Pipeline

A modular PyTorch-based system to evaluate FP32, FP16, INT8 (PTQ & QAT)
on DistilBERT using SST-2.

## Features
- FP32 / FP16 / INT8 (PTQ + QAT)
- Accuracy, latency, memory evaluation
- Sensitivity analysis
- Robustness testing
- Gradio demo + Streamlit dashboard

## Run

python scripts/run_benchmark.py
python scripts/run_sensitivity.py

streamlit run app_streamlit.py
python app_gradio.py