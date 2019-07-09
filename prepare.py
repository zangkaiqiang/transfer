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

df_dog = df[df.label == 'dog']
df_cat = df[df.label == 'cat']
df_dog_train = df_dog[:10000]
df_dog_val = df_dog[10000:]
df_cat_train = df_cat[:10000]
df_cat_val = df_cat[10000:]

df_train = df_dog_train.append(df_cat_train)
df_val = df_dog_val.append(df_cat_val)

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,)

train_generator = train_datagen.flow_from_dataframe(dataframe=df_train,
                                                    x_col="filepath",
                                                    y_col="label",
                                                    target_size=(image_size, image_size),
                                                    class_mode="binary",
                                                    batch_sizse=150)

val_generator = train_datagen.flow_from_dataframe(dataframe=df_val,
                                                  x_col="filepath",
                                                  y_col="label",
                                                  target_size=(image_size, image_size),
                                                  class_mode="binary",
                                                  batch_sizse=150)
