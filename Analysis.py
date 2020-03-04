import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import train_test_split

print('loading Drug dataSet...')
encoded_drugs=np.load('drugs_encoded.npy')
print('loading cell dataSet...')
encoded_cells=np.load('cells_encoded.npy')
print('loading IC dataSet...')
encoded_ICs=np.load('ICs_encoded.npy')

print('Spliting Data...')
encoded_drugs_train, encoded_drugs_test,encoded_cells_train, encoded_cells_test, encoded_ICs_train, encoded_ICs_test = train_test_split(encoded_drugs,encoded_cells, encoded_ICs, test_size=0.2)
print('Done')

model=keras.models.load_model('NetworkModel.h5')

predictions=model.predict([encoded_drugs_test,encoded_cells_test])

R2=r2_score(encoded_ICs_test,predictions)
print(R2)
'''
def Pearson(a, b):
    real = tf.squeeze(a)
    pred = tf.squeeze(b)
    real_new = real - tf.reduce_mean(real)
    pred_new = pred - tf.reduce_mean(real)
    up = tf.reduce_mean(tf.multiply(real_new, pred_new))
    real_var = tf.reduce_mean(tf.multiply(real_new, real_new))
    pred_var = tf.reduce_mean(tf.multiply(pred_new, pred_new))
    down = tf.multiply(tf.sqrt(real_var), tf.sqrt(pred_var))
    return tf.div(up, down)
'''
#print(Pearson(encoded_ICs_test,predictions))


#0.6870138806367987
#0.6884151540299246
#0.671154599073001
#0.8066582589785634
#yeeeeeeeessss
#0.8468726257824654
#0.8511071666124208
#SHIT...	
#0.8991339306148485
