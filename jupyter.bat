@echo off
echo 📚 Starting Jupyter Lab...
call conda activate VFM
cd /d "C:\Users\micha\PycharmProjects\VFM"
jupyter lab --no-browser --port=8888
pause
