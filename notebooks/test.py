import numpy as np
import tensorflow as tf
import keras
import utils
import gradio as gr


model = keras.models.load_model(
    "/media/d/skole/dat191/visual-vehicle-recognition-varying-lighting-conditions/models/train_hierarchical_cnn_global_aug_v2.keras")


img = gr.Image(

)


# run prediction
preds = model.predict(val_ds)


# helper to get numpy array from tensor-like objects
def to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)

# print top-k for a single 1D probability array


def top_k_str(probs, class_names=None, k=5):
    probs = np.asarray(probs)
    idx = np.argsort(probs)[::-1][:k]
    lines = []
    for i in idx:
        pct = probs[i] * 100.0
        name = f" ({class_names[i]})" if class_names is not None else ""
        lines.append(f"class {i}{name}: {pct:.2f}%")
    return "; ".join(lines)

# main pretty-printer - handles dict outputs and single arrays


def print_preds(preds, samples=5, top_k=5, class_names_map=None):
    """
    preds: output from model.predict (numpy array, tuple/list, or dict)
    samples: how many samples to print
    top_k: how many top classes to show per level
    class_names_map: optional dict mapping level -> list of class names
    """
    # case: dict of arrays (e.g. {'lvl1': arr1, 'lvl2': arr2})
    if isinstance(preds, dict):
        preds_np = {k: to_numpy(v) for k, v in preds.items()}
        n_samples = next(iter(preds_np.values())).shape[0]
        for s in range(min(samples, n_samples)):
            print(f"Sample {s}:")
            for lvl, arr in preds_np.items():
                names = None
                if class_names_map is not None and lvl in class_names_map:
                    names = class_names_map[lvl]
                print(
                    f"  {lvl}: {top_k_str(arr[s], class_names=names, k=top_k)}")
            print()
        return

    # case: tuple/list of arrays (multiple outputs)
    if isinstance(preds, (list, tuple)):
        preds_np = [to_numpy(p) for p in preds]
        n_samples = preds_np[0].shape[0]
        for s in range(min(samples, n_samples)):
            print(f"Sample {s}:")
            for i, arr in enumerate(preds_np):
                print(f"  output[{i}]: {top_k_str(arr[s], k=top_k)}")
            print()
        return

    # case: single array
    arr = to_numpy(preds)
    n_samples = arr.shape[0]
    for s in range(min(samples, n_samples)):
        print(f"Sample {s}: {top_k_str(arr[s], k=top_k)}")


# Example usage:
# If you have class names for 'lvl2', e.g. class_names_map = {'lvl2': ['car','truck',...]}
class_names_map = None
print_preds(preds, samples=10, top_k=10, class_names_map=class_names_map)
