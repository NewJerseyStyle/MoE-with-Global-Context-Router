@echo off
REM Run Token + Global Context MoE Experiments

echo ========================================
echo Token + Global Context MoE Experiments
echo ========================================

cd /d %~dp0

REM Experiment 1: Momentum + Gate (Our main approach)
echo.
echo [1/4] Testing Momentum + Gate...
python evaluate_token_global.py ^
    --context_method momentum ^
    --fusion_method gate ^
    --num_samples 100 ^
    --output_dir ../results

REM Experiment 2: THOR-style
echo.
echo [2/4] Testing THOR + Gate...
python evaluate_token_global.py ^
    --context_method thor ^
    --fusion_method gate ^
    --num_samples 100 ^
    --output_dir ../results

REM Experiment 3: Combined
echo.
echo [3/4] Testing Combined + Gate...
python evaluate_token_global.py ^
    --context_method combined ^
    --fusion_method gate ^
    --num_samples 100 ^
    --output_dir ../results

REM Experiment 4: No gate
echo.
echo [4/4] Testing Momentum + Add...
python evaluate_token_global.py ^
    --context_method momentum ^
    --fusion_method add ^
    --num_samples 100 ^
    --output_dir ../results

REM Compare
echo.
echo ========================================
echo Comparing all results...
echo ========================================
cd ..
python evaluate.py --compare --output_dir ./results

echo.
echo All experiments complete!
pause
