call conda activate base
for /l %%x in (1, 1, 100) do (
    echo ============== %%x
    cargo run
    python scripts/model_trainer.py
)