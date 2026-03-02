from typing import Dict, Optional


def save_hparams(hparams: Dict, repo: str = '.', experiment: Optional[str] = None):
    """Save hyperparameters to an Aim run.

    Safe no-op if `aim` is not installed. Creates a short-lived Run,
    writes the `hparams` mapping to the run under the key 'hparams',
    then closes the run.
    """
    try:
        from aim import Run
    except Exception:
        # Aim not installed or import failed — fail gracefully.
        print("aim not available; skipping hparams save")
        return

    try:
        run = Run(repo=repo, experiment=experiment)
        run["hparams"] = dict(hparams)
        run.close()
    except Exception as e:
        # Don't raise during training; log and continue.
        print(f"Failed saving hparams to Aim: {e}")
