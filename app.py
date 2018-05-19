import click
import urllib.request
import glob
import cv2
from sklearn import linear_model
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import time
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.contrib.layers import flatten

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo('Invoked command: ')
    else:
        click.echo('Invoked model:  %s' % ctx.invoked_subcommand)

@cli.command()
def download():
    urllib.request.urlretrieve("http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip", "images/FullIJCNN2013.zip")

def read_data():
    images_directory_path = "./images/"
    x_train_names = []
    x_test_names = []
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(43):
        if(i < 10):
            class_path_train = images_directory_path + "train/0" + str(i) + "/*.ppm"
            class_path_test = images_directory_path + "test/0" + str(i) + "/*.ppm"
        else:
            class_path_train = images_directory_path + "train/" + str(i) + "/*.ppm"
            class_path_test = images_directory_path + "test/" + str(i) + "/*.ppm"

        images_path_train = glob.glob(class_path_train)
        images_path_test = glob.glob(class_path_test)
        x_train_names.extend(images_path_train)
        x_test_names.extend(images_path_test)

    for image_path in x_train_names:
        image = cv2.resize(cv2.imread(image_path), (32,32))
        x_train.append(image)
        y_train.append(int(image_path.split("/")[-2]))

    for image_path in x_test_names:
        image = cv2.resize(cv2.imread(image_path), (32,32))
        x_test.append(image)
        y_test.append(int(image_path.split("/")[-2]))

    x_train = np.asarray(x_train)
    x_train = x_train.flatten().reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
    x_test = np.asarray(x_test)
    x_test = x_test.flatten().reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])
    y_test = np.asarray(y_test)
    y_train = np.asarray(y_train)
    return x_train, x_test, y_train, y_test

########### logistic regression with scikit learn ###################
@cli.command()
def regression():
    path_model = "modelScikit.pkl"
    x_train, x_test, y_train, y_test = read_data()
    train_scikit_model(x_train, y_train, path_model)
    test_model(x_test, y_test, path_model)

def train_scikit_model(x, y, path_model):
    t = time.time()
    logreg = linear_model.LogisticRegression()
    logreg.fit(x, y)
    joblib.dump(logreg, path_model)
    print("trainning time: %.4f secs" % (time.time()-t))

def test_model(x, y, path_model):
    model = joblib.load(path_model)
    preds = []
    for i in range(y.size):
        preds.append(model.predict(x[i].reshape(1, -1)))
    preds = np.asarray(preds)
    y = y.reshape(-1,1)
    goods = len(preds[preds==y])
    print("model accuracy: %.4f%%" % ((goods / y.size)*100))

############# logistic regression with tensor flow ###################
@cli.command()
def regression_tensor():
    path_model = "modelTensor.pkl"
    x_train, x_test, y_train, y_test = read_data()
    train_tensor_model(x_train, y_train, x_test, y_test)

def train_tensor_model(X_train, y_train, X_test, y_test):
    # Variables
    W = tf.Variable(tf.random_normal(shape=[X_train.shape[1], 1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Placeholders
    data = tf.placeholder(dtype = tf.float32, shape=[None, X_train.shape[1]])
    target = tf.placeholder(dtype = tf.float32, shape=[None, 1])

    # Model
    model = tf.matmul(data, W) + b

    # Loss function
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels=target))

    # Parameters
    learning_rate = 0.03
    batch_size = 30
    iter_num = 1500

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    goal = optimizer.minimize(loss)

    # Indicators
    prediction = tf.round(tf.sigmoid(model))
    correct = tf.cast(tf.equal(prediction, target), dtype=tf.float32)
    accuracy = tf.reduce_mean(correct)

    # Record
    loss_trace = []
    train_acc = []
    test_acc = []

    for epoch in range(iter_num):
        # Generate random batch index
        batch_index = np.random.choice(len(X_train), size=batch_size)
        batch_x_train = X_train[batch_index]
        batch_y_train = np.matrix(y_train[batch_index]).T
        sess.run(goal, feed_dict={data: batch_x_train, target: batch_y_train})
        temp_loss = sess.run(loss, feed_dict={data: batch_x_train, target: batch_y_train})
        # convert into a matrix, and the shape of the placeholder to correspond
        temp_train_acc = sess.run(accuracy, feed_dict={data: X_train, target: np.matrix(y_train).T})
        temp_test_acc = sess.run(accuracy, feed_dict={data: X_test, target: np.matrix(y_test).T})
        # recode the result
        loss_trace.append(temp_loss)
        train_acc.append(temp_train_acc)
        test_acc.append(temp_test_acc)
        # output
        if (epoch + 1) % 300 == 0:
            print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_loss, temp_train_acc, temp_test_acc))

    # plot loss
    plt.plot(loss_trace)
    plt.title('Cross Entropy Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    # plot accuracy
    plt.plot(train_acc, 'b-', label='train accuracy')
    plt.plot(test_acc, 'k-', label='test accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Train and Test Accuracy')
    plt.legend(loc='best')
    plt.show()


################# model tensor flow lenet 5 ###################
@cli.command()
def lenet5():
    X_train, X_test, y_train, y_test = read_images()
    train_lenet(X_train, X_test, y_train, y_test)

def read_images():
    images_directory_path = "./images/"
    images_train_names = []
    images_test_names = []
    for i in range(43):
        if i < 10:
            images_train_names.extend(glob.glob(images_directory_path + "train/0" + str(i) + "/*.ppm"))
            images_test_names.extend(glob.glob(images_directory_path + "test/0" + str(i) + "/*.ppm"))
        else:
            images_train_names.extend(glob.glob(images_directory_path + "train/" + str(i) + "/*.ppm"))
            images_test_names.extend(glob.glob(images_directory_path + "test/" + str(i) + "/*.ppm"))

    images_train = []
    labels_train = []
    images_test = []
    labels_test = []

    for path in images_train_names:
        image = cv2.resize(cv2.imread(path, 0), (32,32))
        images_train.append(image)
        labels_train.append(int(path.split("/")[-2]))

    for path in images_test_names:
        image = cv2.resize(cv2.imread(path, 0), (32,32))
        images_test.append(image)
        labels_test.append(int(path.split("/")[-2]))

    images_train = np.asarray(images_train)[:,:,:,np.newaxis].astype(np.float32)
    labels_train = np.asarray(labels_train)
    images_test = np.asarray(images_test)[:,:,:,np.newaxis].astype(np.float32)
    labels_test = np.asarray(labels_test)

    return images_train, images_test, labels_train, labels_test

def evaluate(X_data, y_data, BATCH_SIZE, accuracy, X, y):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy_ = sess.run(accuracy, feed_dict={X: batch_x, y: batch_y})
        total_accuracy += (accuracy_ * len(batch_x))
    return total_accuracy / num_examples

def train_lenet(X_train, X_test, y_train, y_test):
    tf.reset_default_graph()
    tf.set_random_seed(10)

    X = tf.placeholder(tf.float32, (None, X_train.shape[1], X_train.shape[1], 1), name='X')
    y = tf.placeholder(tf.int32, (None), name='Y')
    y_one_hot = tf.one_hot(y, 43)

    #################### layer 1 ####################
    W1 = tf.get_variable('W1', (5, 5, 1, 6), initializer = tf.truncated_normal_initializer())
    b1 = tf.get_variable('b1', initializer = np.zeros(6, dtype=np.float32))
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='VALID') + b1
    A1 = tf.nn.relu(Z1)
    A1max_pool = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


    #################### layer 2 ####################
    W2 = tf.get_variable('W2', (5, 5, 6, 16), initializer=tf.truncated_normal_initializer())
    b2 = tf.get_variable('b2', initializer=np.zeros(16, dtype=np.float32))
    Z2 = tf.nn.conv2d(A1max_pool, W2, strides=[1, 1, 1, 1], padding='VALID') + b2
    A2 = tf.nn.relu(Z2)
    A2max_pool = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    A2flat = flatten(A2max_pool)

    #################### layer 3 ####################
    W3 = tf.get_variable("W3", (400, 120), initializer = tf.truncated_normal_initializer()) # for 80
    # W3 = tf.get_variable("W3", (7744, 120), initializer = tf.truncated_normal_initializer()) # for 100
    # W3 = tf.get_variable("W3", (150544, 120), initializer = tf.truncated_normal_initializer()) # for 400
    b3 = tf.get_variable("b3", initializer = np.zeros(120, dtype=np.float32))
    Z3 = tf.add(tf.tensordot(A2flat, W3, [[1], [0]]), b3)
    A3 = tf.nn.relu(Z3)

    #################### layer 4 ####################
    W4 = tf.get_variable("W4", (120, 84), initializer = tf.truncated_normal_initializer())
    b4  = tf.get_variable("b4", initializer = np.zeros(84, dtype=np.float32))
    Z4 = tf.add(tf.tensordot(A3, W4, [[1], [0]]), b4)
    A4 = tf.nn.relu(Z4)

    #################### output layer ####################
    W_l = tf.get_variable("W_l", (84, 43), initializer=tf.truncated_normal_initializer())
    b_l = tf.get_variable("b_l", initializer=np.zeros(43, dtype=np.float32))

    logits = tf.add(tf.tensordot(A4, W_l, [[1], [0]]), b_l)

    #################### trainning ####################
    entropy   = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_one_hot)
    loss      = tf.reduce_mean(entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train     = optimizer.minimize(loss)
    correct   = tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1))
    accuracy  = tf.reduce_mean(tf.cast(correct, tf.float32))

    saver = tf.train.Saver()

    BATCH_SIZE = 64
    EPOCHS = 300
    EVALUATE_EVERY_N_EPOCHS = 5

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        t0 = time.time()
        for epoch in range(EPOCHS):

            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_X = X_train[offset:end]
                batch_y = y_train[offset:end]
                sess.run(train, feed_dict={X: batch_X, y: batch_y})

            if (epoch % EVALUATE_EVERY_N_EPOCHS) == 0:
                train_accuracy = evaluate(X_train, y_train, BATCH_SIZE, accuracy, X, y)
                validation_accuracy = evaluate(X_test, y_test, BATCH_SIZE, accuracy, X, y)
                fortmat_string = "EPOCH({})\t -> Train Accuracy = {:.3f} | Validation Accuracy = {:.3f}"
                print(fortmat_string.format(epoch, train_accuracy, validation_accuracy))
        t1 = time.time()
        total = t1-t0
        print("trainning elapsed time", round(total, 2), "seconds")
        saver.save(sess, './lenet-5')


if __name__ == '__main__':
    cli()
