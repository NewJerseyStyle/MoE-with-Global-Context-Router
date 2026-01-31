@echo off
REM Run All MoE Routing Experiments
REM Methods:
REM   1. Linear Baseline (mergekit static merge)
REM   2. Model-Level Router (MoDEM-style)
REM   3. Token + Global Context - Untrained
REM   4. Token + Global Context - Trained

echo ========================================
echo MoE Routing Experiment - Full Suite
echo ========================================
echo.
echo Methods to compare:
echo   1. Linear (static merge)
echo   2. MoDEM (model-level routing)
echo   3. Token+Global Untrained
echo   4. Token+Global Trained
echo.

cd /d %~dp0

set NUM_SAMPLES=100
set TRAIN_SAMPLES=500
set TRAIN_EPOCHS=3

REM ========================================
REM Method 1: Linear Baseline (mergekit)
REM ========================================
echo [1/5] LINEAR BASELINE
echo ========================================

cd linear_baseline

if not exist "merged_linear\config.json" (
    echo Merging models with mergekit...
    mergekit-yaml merge_config.yaml .\merged_linear --cuda --low-cpu-memory
) else (
    echo Merged model exists, skipping merge.
)

cd ..
python evaluate.py --model_path ./linear_baseline/merged_linear --method_name linear --num_samples %NUM_SAMPLES%

REM ========================================
REM Method 2: Model-Level Router (MoDEM)
REM ========================================
echo.
echo [2/5] MODEL-LEVEL ROUTER (MoDEM)
echo ========================================

cd model_router
python evaluate_model_router.py ^
    --compute_centroids ^
    --num_samples %NUM_SAMPLES% ^
    --output_dir ../results

cd ..

REM ========================================
REM Method 3: Token + Global (Untrained)
REM ========================================
echo.
echo [3/5] TOKEN + GLOBAL CONTEXT (Untrained)
echo ========================================

cd token_global_router

python evaluate_token_global.py ^
    --context_method momentum ^
    --fusion_method gate ^
    --num_samples %NUM_SAMPLES% ^
    --output_dir ../results

cd ..

REM ========================================
REM Method 4: Train Token + Global Router
REM ========================================
echo.
echo [4/5] TRAINING TOKEN + GLOBAL ROUTER
echo ========================================

cd token_global_router

if not exist "trained_router\final\config.json" (
    echo Training router...
    python train_router.py ^
        --num_epochs %TRAIN_EPOCHS% ^
        --batch_size 2 ^
        --gradient_accumulation_steps 8 ^
        --learning_rate 1e-4 ^
        --num_samples_per_domain %TRAIN_SAMPLES% ^
        --max_length 512 ^
        --context_method momentum ^
        --fusion_method gate ^
        --load_balance_weight 0.01 ^
        --output_dir ./trained_router
) else (
    echo Trained router exists, skipping training.
)

cd ..

REM ========================================
REM Method 5: Token + Global (Trained)
REM ========================================
echo.
echo [5/5] TOKEN + GLOBAL CONTEXT (Trained)
echo ========================================

cd token_global_router

python evaluate_token_global.py ^
    --context_method momentum ^
    --fusion_method gate ^
    --checkpoint_path ./trained_router/final ^
    --num_samples %NUM_SAMPLES% ^
    --output_dir ../results

cd ..

REM ========================================
REM Final Comparison
REM ========================================
echo.
echo ========================================
echo FINAL COMPARISON
echo ========================================

python evaluate.py --compare --output_dir ./results

echo.
echo ========================================
echo All experiments complete!
echo ========================================
echo.
echo Results saved in: ./results/
echo.
echo Summary of methods:
echo   - linear: Static weight merge (no routing)
echo   - model_router: MoDEM-style model-level routing
echo   - token_global_momentum_gate: Token+Global (untrained)
echo   - token_global_momentum_gate_trained: Token+Global (trained)
echo ========================================
pause
