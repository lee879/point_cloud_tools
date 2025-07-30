@echo off
REM 自动安装依赖并运行主程序

REM 1. 检查 requirements.txt 是否存在
IF NOT EXIST requirements.txt (
    echo [错误] 未找到 requirements.txt 文件，请先生成或添加依赖清单。
    pause
    exit /b
)

REM 2. 安装依赖
echo 正在安装依赖包...
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    echo [错误] 依赖安装失败，请检查 Python 和 pip 是否已正确安装。
    pause
    exit /b
)

REM 3. 运行主程序
echo 正在运行 main.py ...
python main.py
IF %ERRORLEVEL% NEQ 0 (
    echo [错误] main.py 运行失败，请检查代码和依赖。
    pause
    exit /b
)

pause
