#!/usr/bin/env python

from pathlib import Path
import keras
from keras.layers import Rescaling

INPUT_MODEL_PATH = "./models/augment.keras"
OUTPUT_MODEL_PATH = "./models/augment_no_rescale.keras"


def main():
    full_model = keras.models.load_model(INPUT_MODEL_PATH, compile=False)
    full_model.summary()

    # Find Rescaling layer
    rescale_layer = None
    for layer in full_model.layers:
        if isinstance(layer, Rescaling):
            rescale_layer = layer
            break

    if rescale_layer is None:
        raise RuntimeError("No Rescaling layer found in the loaded model.")

    print(f"Found Rescaling layer: {rescale_layer.name}")

    new_input = rescale_layer.output

    # Get named outputs from the original model
    lvl1_out = full_model.get_layer("lvl1").output
    lvl2_out = full_model.get_layer("lvl2").output

    # Preserve dict-style outputs
    model_no_rescale = keras.Model(
        inputs=new_input,
        outputs={"lvl1": lvl1_out, "lvl2": lvl2_out},
        name=full_model.name + "_no_rescale",
    )

    model_no_rescale.summary()

    Path(OUTPUT_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    model_no_rescale.save(OUTPUT_MODEL_PATH)
    print(f"Saved model without Rescaling to {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()
