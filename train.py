import tensorflow as tf
import tensorflow.keras as k
import numpy as np
from load_and_augment import load_and_augment_data
from modelconfig import modelconfig
from compile_model import compile_model_adam
import compile_model

path=r'/content/drive/My Drive/data'
testing_path=r'/content/drive/My Drive/test/'
training_gen,val_gen,test_gen=load_and_augment_data(path,testing_path)
model=modelconfig(0.25)
model=compile_model_adam(model,0.0001,1.2)
cb=tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=4,
                              verbose=0, mode='auto')
history=model.fit_generator(generator=training_gen,steps_per_epoch=25,epochs=100,validation_data=val_gen, validation_steps=10,callbacks=[cb])
training=pd.DataFrame(history.history)
training.to_csv('training_statistics.csv',index=False)
evaluation_test=model.evaluate_gen(test_gen)
print('test accuracy= {} and f1={}'.format(evaluation_test[1],evaluation_test[2]))
model.save('model_polythene.h5')
