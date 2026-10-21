"""Train the model.  Run with:  python train.py  (add --fast for a quick smoke run)."""
import sys

from heart_disease.train import main

if __name__ == "__main__":
    sys.exit(main())
