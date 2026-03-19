from __future__ import annotations

import tensorflow as tf


def center_crop_after_resize_short_side(img: tf.Tensor, target: int) -> tf.Tensor:
    """
    Bevarer aspect ratio, skalerer slik at korteste side = target,
    og center-cropper deretter til target x target.
    """
    shape = tf.shape(img)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)

    scale = tf.cast(target, tf.float32) / tf.minimum(h, w)
    new_h = tf.cast(tf.math.ceil(h * scale), tf.int32)
    new_w = tf.cast(tf.math.ceil(w * scale), tf.int32)

    img = tf.image.resize(img, [new_h, new_w],
                          antialias=True, method="bilinear")
    img = tf.image.resize_with_crop_or_pad(img, target, target)
    return img


def hybrid_crop_reflect_landscape(
    img: tf.Tensor,
    target: int,
    crop_fraction_of_excess: float = 0.5,
) -> tf.Tensor:
    """
    Landscape-hybrid:
    1) Skaler slik at høyden = target
    2) Crop bort en andel av overskuddsbredden
    3) Reskaler hele bildet slik at bredden = target
    4) Reflect-pad høyden opp til target
    """
    shape = tf.shape(img)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)

    # Steg 1: skaler slik at høyden blir target
    scale = tf.cast(target, tf.float32) / h
    new_h = tf.cast(target, tf.int32)
    new_w = tf.cast(tf.math.ceil(w * scale), tf.int32)
    new_w = tf.maximum(new_w, 1)

    img = tf.image.resize(img, [new_h, new_w],
                          antialias=True, method="bilinear")

    # Steg 2: crop bort en andel av overskuddsbredden
    excess_w = tf.maximum(new_w - target, 0)

    crop_total = tf.cast(
        tf.round(tf.cast(excess_w, tf.float32) * crop_fraction_of_excess),
        tf.int32,
    )

    crop_left = crop_total // 2
    crop_right = crop_total - crop_left

    img = tf.cond(
        crop_total > 0,
        lambda: img[:, crop_left:new_w - crop_right, :],
        lambda: img,
    )

    cropped_shape = tf.shape(img)
    cropped_h = cropped_shape[0]
    cropped_w = cropped_shape[1]

    # Steg 3: reskaler slik at bredden blir target
    scale2 = tf.cast(target, tf.float32) / tf.cast(cropped_w, tf.float32)
    resized_h = tf.cast(
        tf.math.round(tf.cast(cropped_h, tf.float32) * scale2),
        tf.int32,
    )
    resized_h = tf.maximum(resized_h, 1)

    img = tf.image.resize(img, [resized_h, target],
                          antialias=True, method="bilinear")

    # Steg 4: reflect-pad høyden opp til target
    pad_h = target - resized_h
    top = pad_h // 2
    bottom = pad_h - top

    can_reflect = tf.logical_and(top < resized_h, bottom < resized_h)

    img = tf.cond(
        pad_h > 0,
        lambda: tf.cond(
            can_reflect,
            lambda: tf.pad(
                img, [[top, bottom], [0, 0], [0, 0]], mode="REFLECT"),
            lambda: tf.pad(img, [[top, bottom], [0, 0], [0, 0]],
                           mode="CONSTANT", constant_values=0.0),
        ),
        lambda: img,
    )

    return img


def decode_image_file(path: tf.Tensor) -> tf.Tensor:
    """
    Leser JPEG eller PNG og returnerer RGB-bilde som float32 i [0, 1].
    """
    lower = tf.strings.lower(path)
    img_bytes = tf.io.read_file(path)

    is_png = tf.strings.regex_full_match(lower, ".*\\.png")

    img = tf.cond(
        is_png,
        lambda: tf.image.decode_png(img_bytes, channels=3),
        lambda: tf.image.decode_jpeg(img_bytes, channels=3),
    )

    return tf.image.convert_image_dtype(img, tf.float32)


def preprocess_image_tensor(
    img: tf.Tensor,
    target: int,
    crop_fraction_of_excess: float = 0.5,
) -> tf.Tensor:
    """
    Tar inn et ferdig bilde som tensor og bruker formatavhengig preprocessing:
    - square/portrait -> resize + center crop
    - landscape -> hybrid crop + reflect

    Forventet input:
    - shape [H, W, 3]
    - dtype uint8 eller float32
    - RGB-kanaler
    """
    img = tf.convert_to_tensor(img)

    # Konverter til float32 i [0, 1] hvis nødvendig
    if img.dtype != tf.float32:
        img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.shape(img)
    h = shape[0]
    w = shape[1]

    img = tf.cond(
        h >= w,
        lambda: center_crop_after_resize_short_side(img, target),
        lambda: hybrid_crop_reflect_landscape(
            img,
            target,
            crop_fraction_of_excess=crop_fraction_of_excess,
        ),
    )

    img = tf.ensure_shape(img, [target, target, 3])
    return img


def decode_and_preprocess(
    path: tf.Tensor,
    target: int,
    crop_fraction_of_excess: float = 0.5,
) -> tf.Tensor:
    """
    Leser bildefil fra path, dekoder til RGB float32 [0,1],
    og bruker samme preprocessing som preprocess_image_tensor().
    """
    img = decode_image_file(path)
    return preprocess_image_tensor(
        img,
        target=target,
        crop_fraction_of_excess=crop_fraction_of_excess,
    )
