@echo off
echo 📊 Evaluating VFM Model...
call conda activate VFM
cd /d "C:\Users\micha\PycharmProjects\VFM"
python scripts\evaluate.py experiment=mvp_demo
if %ERRORLEVEL% EQU 0 (
    echo ✅ Evaluation completed successfully!
) else (
    echo ❌ Evaluation failed with error %ERRORLEVEL%
)
pause
