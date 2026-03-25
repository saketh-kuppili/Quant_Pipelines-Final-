from setuptools import setup, find_packages

setup(
    name="quant_pipeline",
    version="0.1.0",
    description="Layer-Aware Quantization Pipeline for DistilBERT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "scikit-learn",
        "numpy",
        "tqdm",
        "matplotlib",
        "pandas",
    ],
    extras_require={
        "ui": ["streamlit", "gradio"],
        "dev": ["pytest"],
    },
)
