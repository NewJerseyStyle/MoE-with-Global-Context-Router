@echo off
REM Train Token + Global MoE Router

echo ========================================
echo Training Token + Global MoE Router
echo ========================================

cd /d %~dp0

REM Full training
python train_router.py ^
    --num_epochs 3 ^
    --batch_size 2 ^
    --gradient_accumulation_steps 8 ^
    --learning_rate 1e-4 ^
    --num_samples_per_domain 500 ^
    --max_length 512 ^
    --context_method momentum ^
    --fusion_method gate ^
    --load_balance_weight 0.01 ^
    --output_dir ./trained_router

echo.
echo Training complete! Now run evaluation:
echo   python evaluate_token_global.py --checkpoint_path ./trained_router/final
pause
