@echo off
echo 🧪 Running Tests...
call conda activate VFM
cd /d "C:\Users\micha\PycharmProjects\VFM"
python -m pytest tests/ -v --cov=src/vfm
pause
