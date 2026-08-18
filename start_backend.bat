@echo off
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
pause
