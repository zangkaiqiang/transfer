import os
import pandas as pd
from tensorflow import keras
from model import image_size

filename = os.listdir('input/train')

df = pd.DataFrame({'filepath': filename})
df['filepath'] = df.filepath.apply(lambda x: os.path.join('input/train', x))


def label_judge(s):
    if 'dog' in s:
        return 'dog'
    else:
        return 'cat'


df['label'] = df.filepath.apply(lambda x: label_judge(x))

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1)

train_generator = train_datagen.flow_from_dataframe(dataframe=df,
                                                    x_col="filepath",
                                                    y_col="label",
                                                    target_size=(image_size, image_size),
                                                    class_mode="binary",
                                                    batch_sizse=150)

# modelHistory = model.fit_generator(train_datagenerator, epochs=50)


