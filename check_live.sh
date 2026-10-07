#!/bin/bash
# Check if the live dashboard starts successfully
set -e
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi
streamlit run app.py --server.headless true --server.port 8501 --server.runOnSave false
