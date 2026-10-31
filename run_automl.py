
from src.automl import run_automl

if __name__ == "__main__":
    run_automl(n_trials=30, use_early_features=True)
