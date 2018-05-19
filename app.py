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

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo('Invoked command: ')
    else:
        click.echo('Invoked model:  %s' % ctx.invoked_subcommand)

@cli.command()
def download():
    #urllib.request.urlretrieve ("https://t2.uc.ltmcdn.com/images/5/0/6/img_como_saber_si_un_gato_es_macho_o_hembra_con_fotos_10605_600.jpg", "images/cat.jpg")
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
    #images_names = read_images()
    x_train, x_test, y_train, y_test = read_data()
    train_scikit_model(x_train, y_train, path_model)
    test_model(x_test, y_test, path_model)

    # model = joblib.load(path_model)
    # testeo  = cv2.resize(cv2.imread("./images/FullIJCNN2013/40/00000.ppm"), (32,32))
    # testeo = testeo.flatten().reshape(1, 32*32*3)
    # #print(model.predict(x_test[0].reshape(1, -1)))
    # print(model.predict(testeo))

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
def regression2():
    path_model = "modelTensor.pkl"
    x_train, x_test, y_train, y_test = read_data()
    train_tensor_model(x_train, y_train, x_test, y_test)

def train_tensor_model(X_train, y_train, X_test, y_test):
    # # Parameters
    # learning_rate = 0.001
    # n_epochs = 3000
    # display_step = 500
    #
    # X = tf.placeholder(tf.float32, [None, X_train.shape[1]])
    # y = tf.placeholder(tf.float32, [None, 43])
    #
    # # Variables
    # W = tf.Variable(tf.random_normal([X_train.shape[1], 43]))
    # b = tf.Variable(tf.random_normal([43]))
    #
    # y_ = tf.nn.softmax(tf.add(tf.matmul(X, W), b))
    #
    # one_hot_encoder = OneHotEncoder()
    #
    # one_hot_encoder.fit(y_train.reshape(-1, 1))
    #
    # y_one_hot = one_hot_encoder.transform(y_train.reshape(-1, 1))#.reshape(len(y_train), 43)
    # print(X_train.shape)
    # print(y_one_hot.shape)
    #
    # #y_one_hot = tf.one_hot(y_train, 43)
    #
    # cost = tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    #
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # for epoch in range(n_epochs):
    #     cost_in_each_epoch = 0
    #
    #     _, c = sess.run([optimizer, cost], feed_dict={X: X_train, y: y_one_hot})
    #     cost_in_each_epoch += c
    #
    #     if (epoch+1) % display_step == 0:
    #         print("Epoch: {}".format(epoch + 1), "cost={}".format(cost_in_each_epoch))
    #
    # correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print("Accuracy:", accuracy.eval({X: X_test, y: y_test}))



    # seed = 7
    # np.random.seed(seed)
    # tf.set_random_seed(seed)
    #
    # # Variables
    # W = tf.Variable(tf.random_normal(shape=[x_train.shape[1], 1]))
    # b = tf.Variable(tf.random_normal(shape=[x_train.shape[1], 1]))
    #
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    #
    # # Placeholders
    # x = tf.placeholder(dtype=tf.float32, shape=[None, x_train.shape[1]])
    # y_ = tf.placeholder(dtype=tf.int32, shape=[None, 43])
    #
    # # Model
    # model = tf.add(tf.matmul(x, W), b)
    # #out = tf.nn.softmax(model)
    #
    # logits = tf.one_hot(tf.cast(model, tf.int32), 43)
    # y_one_hot = tf.one_hot(y_, 43)
    #
    # # Loss function
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= y_one_hot))
    #
    # # Parameters
    # learning_rate = 0.03
    # n_iters = 1500
    #
    # # Optimizer
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # train = optimizer.minimize(loss)
    # correct = tf.equal(tf.argmax(logits,1), tf.argmax(y_one_hot, 1))
    # accuracy = tf.reduce_mean(correct)
    #
    # for epoch in range(n_iters):
    #     sess.run(train, feed_dict={x: x_train, y_:y_one_hot})
    #     error = sess.run(loss, feed_dict={x: x_train, y_:y_train})
    #     train_accuracy = sess.run(accuracy, feed_dict={x: x_train, y_: y_train})
    #     test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
    #     if (epoch + 1) % 300 == 0:
    #         print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, error, train_accuracy, test_accuracy))



    #Variables
    W = tf.Variable(tf.random_normal(shape=[x_train.shape[1], 1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    #place holders
    data = tf.placeholder(dtype = tf.float32, shape=[None, x_train.shape[1]])
    target = tf.placeholder(dtype = tf.float32, shape=[None, 1])

    #model
    model = tf.matmul(data, W) + b

    #loss function
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels=target))

    #parameters
    learning_rate = 0.03
    batch_size = 30
    iter_num = 1500

    #optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    goal = optimizer.minimize(loss)

    #indicators
    prediction = tf.round(tf.sigmoid(model))
    correct = tf.cast(tf.equal(prediction, target), dtype=tf.float32)
    accuracy = tf.reduce_mean(correct)

    #record
    loss_trace = []
    train_acc = []
    test_acc = []

    for epoch in range(iter_num):
        # Generate random batch index
        batch_index = np.random.choice(len(x_train), size=batch_size)
        batch_x_train = x_train[batch_index]
        batch_y_train = np.matrix(y_train[batch_index]).T
        sess.run(goal, feed_dict={data: batch_x_train, target: batch_y_train})
        temp_loss = sess.run(loss, feed_dict={data: batch_x_train, target: batch_y_train})
        # convert into a matrix, and the shape of the placeholder to correspond
        temp_train_acc = sess.run(accuracy, feed_dict={data: x_train, target: np.matrix(y_train).T})
        temp_test_acc = sess.run(accuracy, feed_dict={data: x_test, target: np.matrix(y_test).T})
        # recode the result
        loss_trace.append(temp_loss)
        train_acc.append(temp_train_acc)
        test_acc.append(temp_test_acc)
        # output
        if (epoch + 1) % 300 == 0:
            print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_loss, temp_train_acc, temp_test_acc))
    #
    # plt.plot(loss_trace)
    # plt.title('Cross Entropy Loss')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.show()

    # accuracy
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
    x_train, x_test, y_train, y_test = read_images()

def read_images():
    images_directory_path = "./images/FullIJCNN2013/"
    images_train_names = []
    images_test_names = []
    for i in range(43):
        if i < 10:
            images_train_names = glob.glob(images_directory_path + "train/0" + str(i) + "/*.ppm")
            images_test_names = glob.glob(images_directory_path + "test/0" + str(i) + "/*.ppm")
        else:
            images_train_names = glob.glob(images_directory_path + "train/" + str(i) + "/*.ppm")
            images_test_names = glob.glob(images_directory_path + "test/" + str(i) + "/*.ppm")

    images_train = []
    images_test = []
    for path in images_train:
        image = cv2.imread(path)

if __name__ == '__main__':
    cli()

# import click
#
# @click.command()
# @click.option('--count', default=1, help='Number of greetings.')
# @click.option('--name', prompt='Your name',
#               help='The person to greet.')
# def hello(count, name):
#     """Simple program that greets NAME for a total of COUNT times."""
#     for x in range(count):
#         click.echo('Hello %s!' % name)
#
# if __name__ == '__main__':
#     hello()
