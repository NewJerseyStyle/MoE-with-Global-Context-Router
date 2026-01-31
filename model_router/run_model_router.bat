@echo off
REM Run Model-Level Router Evaluation

echo ========================================
echo Model-Level Router Experiment
echo ========================================

cd /d %~dp0

REM Compute centroids and evaluate
python evaluate_model_router.py ^
    --compute_centroids ^
    --num_samples 100 ^
    --output_dir ../results ^
    --benchmarks gsm8k mmlu humaneval

echo.
echo Done! Check ../results/ for output.
pause
