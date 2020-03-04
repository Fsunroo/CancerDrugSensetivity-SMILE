print('loading Modules...')
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
print('loading Modules DONE-')


print('loading Drug dataSet...')
encoded_drugs=np.load('drugs_encoded.npy')
print('loading cell dataSet...')
encoded_cells=np.load('cells_encoded.npy')
print('loading IC dataSet...')
encoded_ICs=np.load('ICs_encoded.npy')

print('Spliting Data...')
encoded_drugs_train, encoded_drugs_test,encoded_cells_train, encoded_cells_test, encoded_ICs_train, encoded_ICs_test = train_test_split(encoded_drugs,encoded_cells, encoded_ICs, test_size=0.2)
print('Done')

input1=keras.layers.Input(shape=(140,30,))
x1=keras.layers.Flatten(input_shape=(140,30,))(input1)
x2=keras.layers.Dense(64,activation='relu')(x1)
x3=keras.layers.Dense(64,activation='relu')(x2)

input2=keras.layers.Input(shape=(2,735))
y1=keras.layers.Flatten(input_shape=(2,735,))(input2)
y2=keras.layers.Dense(128,activation='relu')(y1)
y3=keras.layers.Dense(64,activation='relu')(y2)

merged=keras.layers.concatenate([x3,y3],axis=-1)

z=keras.layers.Dense(64,activation='relu')(merged)
out=keras.layers.Dense(1,activation='linear')(z)

model=keras.models.Model(inputs=[input1,input2], outputs=out)

model.compile(optimizer='sgd',loss='mean_squared_error',metrics=['mse'])

model.fit([encoded_drugs_train,encoded_cells_train],encoded_ICs_train,validation_split = 0.2,epochs=200)

model.save('NetworkModel.h5')

