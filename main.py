import numpy as np 
import manip_data
import model
import tensorflow as tf
import pandas as pd


def myprint(s):
    with open('modelsummary.txt','a') as f:
        print(s, file=f)
tf.config.run_functions_eagerly(True)


data_address = "Dataset/image_mixed.npy"
label_address = "Dataset/label_mixed.npy"
data_object = manip_data.Data_Manipulation(data_address, label_address)
train_data, train_label, test_data, test_label = data_object.shuffle_split(0.2)
ldw_model = model.Deep_LDW(8)
stopper_callback = model.Earlystopper()
ldw_model.compile(optimizer='adam', \
				loss='binary_crossentropy', metrics=['accuracy'])
ldw_model.build((None, 80, 160, 3))

ldw_model.model().summary(print_fn=myprint)
ldw_model.save_weights("LDW_Net.h5")
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
validation = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
flowed_datagen = datagen.flow(train_data, train_label, batch_size=32)

flowed_validation = validation.flow(test_data, test_label, batch_size=8)
epochs = 1
history = ldw_model.fit_generator(flowed_datagen, validation_data=flowed_validation, steps_per_epoch=train_data.shape[0]/ 32, epochs=epochs, callbacks=[stopper_callback])
hist_df = pd.DataFrame(history.history) 
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)