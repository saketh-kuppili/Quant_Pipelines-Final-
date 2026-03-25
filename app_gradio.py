"""
Gradio demo for quantized DistilBERT inference.
"""

import gradio as gr
from quant_pipeline.core.pipeline import Pipeline


def predict(text, precision):
    pipe = Pipeline(precision=precision)
    res = pipe.predict(text)

    label_map = {0: "NEGATIVE", 1: "POSITIVE"}
    label = label_map.get(res["label"], str(res["label"]))

    return f"{label} ({res['confidence']:.2f}) | Mode: {res['mode']}"


gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Input Text", placeholder="Enter a sentence..."),
        gr.Dropdown(
            ["fp32", "fp16", "int8_ptq", "int8_qat"],
            value="fp32",
            label="Precision Mode",
        ),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Quantized DistilBERT — Sentiment Analysis",
    description="Compare inference across FP32, FP16, INT8 PTQ, and INT8 QAT modes.",
).launch()