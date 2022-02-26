import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.compat.v1.Session( config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))
X = X/255.0
y= np.array(y)

dense_layers = [0,1,2]
layer_sizes = [32, 64, 128]
conv_layers =[1,2,3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = f"{conv_layer}-convs-{layer_size}-nodes-{dense_layer}-dense{int(time.time())}"
            tensorboard = TensorBoard(log_dir=f"log/{NAME}")
            model = Sequential()

            model.add(Conv2D(layer_size,(3,3),input_shape = X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for i in range(conv_layer-1): 
            
                model.add(Conv2D(layer_size,(3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())

            for i in range(dense_layer):
                model.add(Dense(dense_layer))
                model.add(Activation("relu"))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
            model.fit(X,y,batch_size=32,epochs=5,validation_split=0.1,callbacks=[tensorboard])
