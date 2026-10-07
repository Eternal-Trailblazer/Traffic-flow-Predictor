# Check if the live dashboard starts successfully
set -e
.venv\Scripts\activate
streamlit run app.py --server.headless true --server.port 8501 --server.runOnSave false
pause
