@echo off
echo 🚀 Training VFM Model...
call conda activate VFM
cd /d "C:\Users\micha\PycharmProjects\VFM"
python scripts\train.py experiment=mvp_demo
if %ERRORLEVEL% EQU 0 (
    echo ✅ Training completed successfully!
) else (
    echo ❌ Training failed with error %ERRORLEVEL%
)
pause
