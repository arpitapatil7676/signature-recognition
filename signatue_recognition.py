import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Dense, Activation, Lambda
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from PIL import Image

ROWS = 190
COLS = 160
CHANNELS = 3

TRAIN_DIR = "C:/Users/Admin/Downloads/Signature-recognition-master/Signature-recognition-master/extracted/CEDAR/CEDAR/"
TEST_DIR  = "C:/Users/Admin/Downloads/Signature-recognition-master/Signature-recognition-master/test/"

# Use first 12 classes from CEDAR
SIGNATURE_CLASSES = [str(i) for i in range(1, 13)]

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def get_images(cls):
    cls_dir = os.path.join(TRAIN_DIR, cls)
    return [cls + '/' + im for im in os.listdir(cls_dir) if im.endswith('.png')]

def read_image(src):
    im = Image.open(src).convert('RGB').resize((COLS, ROWS))
    return np.array(im)

files = []
y_all = []

for cls in SIGNATURE_CLASSES:
    cls_files = get_images(cls)
    files.extend(cls_files)
    y_all.extend(np.tile(cls, len(cls_files)))
    print("{0} photos of class {1}".format(len(cls_files), cls))

y_all = np.array(y_all)
print("Total images:", len(files))

X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)
for i, im in enumerate(files):
    X_all[i] = read_image(TRAIN_DIR + im)
    if i % 100 == 0:
        print('Processed {} of {}'.format(i, len(files)))

print(X_all.shape)

y_encoded = LabelEncoder().fit_transform(y_all)
y_cat = to_categorical(y_encoded)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_all, y_cat, test_size=0.1, random_state=23, stratify=y_cat)

def center_normalize(x):
    return (x - K.mean(x)) / K.std(x)

model = Sequential([
    Lambda(center_normalize, input_shape=(ROWS, COLS, CHANNELS)),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    Conv2D(64, (3, 3), padding='valid', activation='relu'),
    ZeroPadding2D(padding=(1, 1)),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.25),

    Conv2D(96, (3, 3), padding='same', activation='relu'),
    Conv2D(96, (3, 3), padding='valid', activation='relu'),
    ZeroPadding2D(padding=(1, 1)),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (2, 2), padding='same', activation='relu'),
    Conv2D(128, (2, 2), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(len(SIGNATURE_CLASSES), activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss=root_mean_squared_error)

early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')

model.fit(X_train, y_train, batch_size=64, epochs=3,
          validation_split=0.1, verbose=1, shuffle=True, callbacks=[early_stopping])

preds = model.predict(X_valid, verbose=1)
print("Validation Log Loss: {}".format(log_loss(y_valid, preds)))

test_files = [im for im in os.listdir(TEST_DIR) if im.endswith('.png')]
test = np.ndarray((len(test_files), ROWS, COLS, CHANNELS), dtype=np.uint8)
for i, im in enumerate(test_files):
    test[i] = read_image(TEST_DIR + im)

test_preds = model.predict(test, verbose=1)
submission = pd.DataFrame(test_preds, columns=SIGNATURE_CLASSES)
submission.insert(0, 'image', test_files)
print(submission.head())

submission.to_csv('C:/Users/Admin/Downloads/Signature-recognition-master/Signature-recognition-master/signatureResults.csv', index=False)
print("Done! Results saved to signatureResults.csv")
