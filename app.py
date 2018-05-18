import click
import urllib.request
import glob
import cv2
from sklearn import linear_model
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import time

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

############3 logistic regression with tensor flow ###################
@cli.command()
def regressionTensor():
    path_model = "modelTensor.pkl"
    x_train, x_test, y_train, y_test = read_data()
    train_tensor_model(x_train, y_train)

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
