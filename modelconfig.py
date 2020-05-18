from tensorflow.keras.applications.xception import Xception
import tensorflow.keras as k


def modelconfig(dropout_rate):
	model=k.models.Sequential()
	model_efficient=Xception(include_top=False,input_shape=(224,224,3),weights=None)
	model.add(k.layers.InputLayer((224,224,3)))
	model.add(model_efficient)
	model.add(k.layers.GlobalMaxPool2D())
	model.add(k.layers.Dense(1024,activation='relu'))
	model.add(k.layers.Dropout(dropout_rate))
	model.add(k.layers.Dense(256,activation='relu'))
	model.add(k.layers.Dropout(dropout_rate))
	model.add(k.layers.Dense(3,activation='softmax'))
	return model