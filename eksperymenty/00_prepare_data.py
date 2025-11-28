import os
import numpy as np

from config import *

for ds_name in DATASETS:
    splits_dir = os.path.join(SPLITS_DIR, ds_name)
    os.makedirs(splits_dir, exist_ok=True)

    X, y = DATASETS[ds_name]

    for s_idx, (train, test) in enumerate(SPLITS.split(X, y)):
        np.savez(os.path.join(splits_dir, f"{s_idx}.npz"), **{
            "X_train": X[train],
            "X_test": X[test],
            "y_train": y[train],
            "y_test": y[test],
        })
