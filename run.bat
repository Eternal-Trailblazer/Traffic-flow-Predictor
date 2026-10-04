@echo off
REM Setup and run the Traffic Flow Predictor

python -m venv .venv

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install dependencies
pip install -r requirements.txt

REM Generate synthetic data
python -m src.mock_stream

REM Train models
python -m src.model

REM Run the Streamlit dashboard
streamlit run app.py

pause
