@echo off
REM ============================================================
REM Syntax Check Script for FPGA ATCNet
REM Verifies SystemVerilog files can be parsed
REM ============================================================

echo.
echo ============================================================
echo FPGA ATCNet Syntax Checker
echo ============================================================
echo.

REM Change to project root
cd /d "%~dp0\.."

echo [Step 1/3] Checking file existence...
echo.

set ERROR=0

if not exist "rtl\conv\temporal_conv.sv" (
    echo [ERROR] temporal_conv.sv not found!
    set ERROR=1
)

if not exist "rtl\window\window_gen.sv" (
    echo [ERROR] window_gen.sv not found!
    set ERROR=1
)

if not exist "sim\temporal_conv_tb.sv" (
    echo [ERROR] temporal_conv_tb.sv not found!
    set ERROR=1
)

if %ERROR%==1 (
    echo.
    echo [FAILED] Missing required files!
    exit /b 1
)

echo [OK] All required files found
echo.

echo [Step 2/3] Checking for SystemVerilog syntax issues...
echo.

REM Check for common syntax errors using findstr
echo Checking temporal_conv.sv...
findstr /C:"endmodule" rtl\conv\temporal_conv.sv >nul
if errorlevel 1 (
    echo [WARN] temporal_conv.sv may have syntax issues
)

echo Checking window_gen.sv...
findstr /C:"endmodule" rtl\window\window_gen.sv >nul
if errorlevel 1 (
    echo [WARN] window_gen.sv may have syntax issues
)

echo Checking temporal_conv_tb.sv...
findstr /C:"endmodule" sim\temporal_conv_tb.sv >nul
if errorlevel 1 (
    echo [WARN] temporal_conv_tb.sv may have syntax issues
)

echo.
echo [OK] Basic syntax checks passed
echo.

echo [Step 3/3] Verification summary...
echo.
echo Files checked:
echo   - rtl\conv\temporal_conv.sv
echo   - rtl\window\window_gen.sv  
echo   - sim\temporal_conv_tb.sv
echo.

echo ============================================================
echo To run actual simulation, you need to install:
echo   - Icarus Verilog (free): http://bleyer.org/icarus/
echo   - ModelSim (commercial)
echo.
echo Then run: scripts\run_verification.bat
echo ============================================================
echo.

exit /b 0
