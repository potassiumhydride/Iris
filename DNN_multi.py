import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #don't show unnecessary error messages

#define parameters
learning_rate = 0.01
batch_size = 20
n_epochs = 5000

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

#unlabeled data for inference/prediction
pred_x = np.asarray([
    [5.1, 3.5, 1.4, 0.2, ],
    [5.9, 3.0, 4.2, 1.5, ],
    [5.7, 2.6, 3.5, 0.2]])

#normalise data
def min_max_normalized(data):
    col_max = np.max(data, axis=0)
    col_min = np.min(data, axis=0)
    return np.divide(data - col_min, col_max - col_min)

train_x = min_max_normalized(train_x)
test_x = min_max_normalized(test_x)
pred_x = min_max_normalized(pred_x)

#Step 2: placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None,4])
Y = tf.placeholder(dtype=tf.int32, shape=[None,1]) #cannot use float

#Step 3: construct model to predict Y
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])

#Step 4: loss function
entropy = tf.losses.sparse_softmax_cross_entropy(labels=Y, logits=model(X))
loss = tf.reduce_mean(entropy)

#Step 5: define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

#Step 6: define accuracy
accuracy = tf.metrics.accuracy(labels = Y, predictions=tf.argmax(model(X), axis=1))
accuracy_mpc = tf.metrics.mean_per_class_accuracy(labels = Y, predictions=tf.argmax(model(X),axis=1),num_classes=3)

#data for visualisation
v_loss = []
v_train_acc = []
v_test_acc = []

#Step 7: start training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer()) #for metric.accuracy's local variables
    n_batches = int(len(train_x) / batch_size)

    # train the model n_epochs times
    for i in range(n_epochs):
        total_loss = 0.0
        total_correct_preds = 0.0
        batch_index = np.random.choice(len(train_x), size=batch_size)
        batch_train_x = train_x[batch_index]
        batch_train_y = np.matrix(train_y[batch_index]).T
        for j in range(n_batches):
            _, loss_batch = sess.run([optimizer, loss], {X: batch_train_x, Y: batch_train_y})
            total_loss += loss_batch

            accuracy_train_batch = sess.run(accuracy, {X: batch_train_x, Y: batch_train_y})
            accuracy_train_batch = accuracy_train_batch[0]
            total_correct_preds += accuracy_train_batch

        accuracy_train = total_correct_preds/n_batches

        #test accuracy
        accuracy_test,pred = sess.run([accuracy,model(X)], {X: test_x, Y: np.matrix(test_y).T}) #no batch for this one
        accuracy_test = accuracy_test[0] #accuracy_test returns a tuple of accuracy and ops so choose [0]

        #test accuracy per class
        accuracy_class = sess.run(accuracy_mpc, {X: test_x, Y: np.matrix(test_y).T})
        accuracy_class = accuracy_class[1] * 100

        v_loss.append(total_loss / n_batches)
        v_train_acc.append(accuracy_train)
        v_test_acc.append(accuracy_test)

        if i%1000 == 0:
            print('epoch: {:4d}, loss: {:4f}, train_acc: {:4f}, test_acc: {:4f}'.format(i, total_loss / n_batches,accuracy_train,accuracy_test))
            # print('accuracy per class:\nIris-setosa: {:.2f}%, Iris-versicolor: {:.2f}%, Iris-virginica: {:.2f}%'.format(
            #     accuracy_class[0], accuracy_class[1], accuracy_class[2])) #uncomment this line if you want to print accuracy per class every epoch

    print('accuracy per class:\nIris-setosa: {:.2f}%, Iris-versicolor: {:.2f}%, Iris-virginica: {:.2f}%'.format(accuracy_class[0],accuracy_class[1],accuracy_class[2]))

    #Step 8: predict/ inference
    prediction_output = sess.run(model(X),{X:pred_x})
    class_names = ['Iris-setosa','Iris-versicolor','Iris-virginica']

    for i, logits in enumerate(prediction_output):
        class_idx = (np.argmax(logits)) #if you use tf. something have to type sess.run as it becomes a tensor. if just numpy then no need
        #class_idx = sess.run(tf.argmax(logits))
        p = sess.run(tf.nn.softmax(logits)[class_idx])
        name = class_names[class_idx]
        print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100 * p))
        
    plt.figure(1)
    plt.plot(v_loss) #no need x axis epoch
    plt.title('Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('DNN_multi_accuracy_loss.png')
    # plt.show() #ends program

    plt.figure(2)
    plt.plot(v_train_acc, 'bo', label='train_acc')
    plt.plot(v_test_acc, 'ro', label='test_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('DNN_multi_accuracy_acc.png')

    plt.show()
