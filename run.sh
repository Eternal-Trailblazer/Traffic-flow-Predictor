#!/bin/bash
# Setup and run the Traffic Flow Predictor

set -e

python3 -m venv .venv

# Activate virtual environment (Windows/Unix compatible detection)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    # Windows
    source .venv/Scripts/activate
else
    # Unix/Linux/macOS
    source .venv/bin/activate
fi

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python -m src.mock_stream

# Train models
python -m src.model

# Run the Streamlit dashboard
streamlit run app.py
