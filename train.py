import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

IMG = 64

def load_data(folder):
    X=[]
    y=[]

    classes=["fight","nonfight"]

    for label,cls in enumerate(classes):
        path=os.path.join(folder,cls)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)

            image=cv2.imread(img_path)
            if image is None:
                continue

            image=cv2.resize(image,(IMG,IMG))

            X.append(image)
            y.append(label)

    return np.array(X)/255.0,np.array(y)


X_train,y_train=load_data("frames_train")
X_test,y_test=load_data("frames_train")

model=models.Sequential([
    layers.Input(shape=(64,64,3)),
    layers.Conv2D(32,3,activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(2,activation='softmax')
])

model.compile(
optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy']
)

model.fit(X_train,y_train,epochs=10)

model.save("fight_model.h5")

print("done")