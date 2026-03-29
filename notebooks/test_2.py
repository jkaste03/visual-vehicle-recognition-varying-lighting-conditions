import keras


model = keras.models.load_model(
    "/media/d/skole/dat191/visual-vehicle-recognition-varying-lighting-conditions/models/train_hierarchical_cnn_global_aug_v2.keras")
img = keras.utils.load_img(
    "/media/d/skole/dat191/visual-vehicle-recognition-varying-lighting-conditions/datasett_preprocessed/val/Egenprodusert/Tesla/dark/IMG_1666.jpg")
img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)
img_array /= 255.0

print(model.predict(img_array))
