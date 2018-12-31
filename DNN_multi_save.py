import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import min_max_normalized #import from utils.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.reset_default_graph()

#Step 1: data
filename = 'data/Iris.csv'
data = pd.read_csv(filename)
data.Species = data.Species.replace(to_replace=['Iris-setosa','Iris-versicolor','Iris-virginica'], value=[0,1,2])
x = data.drop(labels=['Id', 'Species'], axis=1).values
y = data.Species.values

#split into train and test
train_index = np.random.choice(len(x), round(len(x) * 0.8), replace=False) # set replace=False, avoid double sampling
test_index = np.array(list(set(range(len(x))) - set(train_index)))

train_x = x[train_index]
train_y = y[train_index]
test_x = x[test_index]
test_y = y[test_index]

train_x = min_max_normalized(train_x) #imported from utils
test_x = min_max_normalized(test_x)

#define parameters
learning_rate = 0.01
batch_size = 20
n_batches = int(len(train_x) / batch_size)
steps = 50000
n_epochs = int((steps - n_batches)/n_batches)

#Step 2: placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None,4])
Y = tf.placeholder(dtype=tf.int32, shape=[None,1]) #cannot use float

#Step 3: construct model & variables to predict Y
global_step = tf.Variable(0,trainable=False,dtype=tf.int32,name='global_step')
#global step to keep track of the steps and can be restored for later

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])

# Add ops to save and restore all the variables.
saver = tf.train.Saver() #have to come after variables/model

#Step 4: loss function
entropy = tf.losses.sparse_softmax_cross_entropy(labels=Y, logits=model(X))
loss = tf.reduce_mean(entropy)

#Step 5: define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss,global_step=global_step) #need to add global step here to increment it by 1 for each training step

#Step 6: define accuracy
accuracy = tf.metrics.accuracy(labels = Y, predictions=tf.argmax(model(X), axis=1))
accuracy_mpc = tf.metrics.mean_per_class_accuracy(labels = Y, predictions=tf.argmax(model(X),axis=1),num_classes=3)

#data for visualisation
v_loss = []
v_train_acc = []
v_test_acc = []
prediction = []

#Step 7: start training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer()) #for metric.accuracy's local variables

    # check if there is a checkpoint, restore it if there is. if there isn’t, train from the start
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoint_directory/DNN_multi_save'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('checkpoint restored:', ckpt.model_checkpoint_path)

    # train the model epochs
    for i in range(n_epochs+2): #to turn the final step into a checkpoint
        total_loss = 0.0
        total_correct_preds = 0.0
        batch_index = np.random.choice(len(train_x), size=batch_size)
        batch_train_x = train_x[batch_index]
        batch_train_y = np.matrix(train_y[batch_index]).T
        for j in range(n_batches):
            step = global_step.eval() #step is for some reason 1 less than global step #use eval() so you don't have to sess.run global step every time
            _, loss_batch = sess.run([optimizer, loss], {X: batch_train_x, Y: batch_train_y})
            total_loss += loss_batch

            accuracy_train_batch = sess.run(accuracy, {X: batch_train_x, Y: batch_train_y})
            accuracy_train_batch = accuracy_train_batch[0]
            total_correct_preds += accuracy_train_batch

            if (step + 1)%10000 == 0:
                saver.save(sess, 'checkpoint_directory/DNN_multi_save', global_step=global_step)
                print('step: {:4d}, loss: {:4f}'.format((step+1), loss_batch))

        accuracy_train = total_correct_preds / n_batches

        #test accuracy
        accuracy_test,pred = sess.run([accuracy,model(X)], {X: test_x, Y: np.matrix(test_y).T}) #no batch for this one
        accuracy_test = accuracy_test[0] #accuracy_test returns a tuple of accuracy and ops so choose [0]

        #test accuracy per class
        accuracy_class = sess.run(accuracy_mpc, {X: test_x, Y: np.matrix(test_y).T})
        accuracy_class = accuracy_class[1] * 100

        v_loss.append(total_loss / n_batches)
        v_train_acc.append(accuracy_train)
        v_test_acc.append(accuracy_test)

    print(
        'train_acc: {:.2f}%, test_acc: {:.2f}%'.format(accuracy_train*100, accuracy_test*100))
    print('accuracy per class:\nIris-setosa: {:.2f}%, Iris-versicolor: {:.2f}%, Iris-virginica: {:.2f}%'.format(
        accuracy_class[0], accuracy_class[1], accuracy_class[2]))

    ##sanity check to see if your accuracies make sense (they should match)
    # print(np.argmax(pred,axis=1))
    # print(test_y)

    plt.figure(1)
    plt.plot(v_loss) #no need x axis epoch
    plt.title('Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('DNN_multi_accuracy_loss.png')

    plt.figure(2)
    plt.plot(v_train_acc, 'bo', label='train_acc')
    plt.plot(v_test_acc, 'ro', label='test_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('DNN_multi_accuracy_acc.png')

    plt.show() #must put it last, otherwise program ends
