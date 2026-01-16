@echo off
REM Quick test with fewer samples to verify setup
echo Quick Test - Verifying Setup (10 samples each)
echo.

cd /d %~dp0

REM Test with base model first (no download needed if cached)
python evaluate.py --model_path Qwen/Qwen3-0.6B --method_name test_base --num_samples 10 --benchmarks gsm8k mmlu

echo.
echo Quick test complete!
pause
