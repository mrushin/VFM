@echo off
echo 🎯 Running VFM Demo...
call conda activate VFM
cd /d "C:\Users\micha\PycharmProjects\VFM"
python scripts\demo.py
if %ERRORLEVEL% EQU 0 (
    echo ✅ Demo completed successfully!
) else (
    echo ❌ Demo failed with error %ERRORLEVEL%
)
pause
