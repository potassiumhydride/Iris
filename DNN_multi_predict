import tensorflow as tf
import numpy as np
import pandas as pd
from utils import min_max_normalized

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.reset_default_graph()

pred_x = np.asarray([
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]])

pred_x = min_max_normalized(prediction_data)

X = tf.placeholder(dtype=tf.float32, shape=[None, 4])

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])

saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoint_directory/DNN_multi_save'))    
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(ckpt.model_checkpoint_path)

    prediction_output = sess.run(model(X), {X: pred_x})
    #print(prediction_output) #sanity check
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    for i, logits in enumerate(prediction_output):
        class_idx = (np.argmax(logits)) 
        # class_idx = sess.run(tf.argmax(logits)) #same
        p = sess.run(tf.nn.softmax(logits)[class_idx])
        name = class_names[class_idx]
        print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100 * p))
