@echo off
REM Linear Merge Script for Windows
REM Prerequisites: pip install mergekit

echo ========================================
echo Linear Merge: Qwen3-0.6B Experts
echo ========================================

set OUTPUT_DIR=.\merged_linear
set CONFIG_FILE=merge_config.yaml

REM Check if mergekit is installed
where mergekit-yaml >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo mergekit not found. Installing...
    pip install mergekit
)

echo.
echo Starting merge...
echo Output directory: %OUTPUT_DIR%
echo.

mergekit-yaml %CONFIG_FILE% %OUTPUT_DIR% --cuda --low-cpu-memory

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================
    echo Merge completed successfully!
    echo Model saved to: %OUTPUT_DIR%
    echo ========================================
) else (
    echo.
    echo Merge failed with error code %ERRORLEVEL%
)

pause
