import os
import numpy as np

from sklearn.base import clone
from rich.progress import Progress, TimeElapsedColumn

from config import *

prog = Progress(
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
)

prog.start()

for ds_name in DATASETS:
    prog_task = prog.add_task(ds_name, total=N_SPLITS * len(METHODS))

    for s_idx in range(N_SPLITS):
        ds = np.load(os.path.join(SPLITS_DIR, ds_name, f"{s_idx}.npz"))
        X_train, y_train, X_test = ds["X_train"], ds["y_train"], ds["X_test"]

        results_dir = os.path.join(PREDICTIONS_DIR, ds_name, str(s_idx))
        os.makedirs(results_dir, exist_ok=True)

        for method_name in METHODS:
            base_classifier = METHODS[method_name]
            clf = clone(base_classifier)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Save results
            np.save(os.path.join(results_dir, f"{method_name}.npy"), y_pred)

            prog.update(prog_task, advance=1)

prog.stop()
