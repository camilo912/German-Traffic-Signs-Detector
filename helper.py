import glob
import cv2
from subprocess import call
#call(["ls", "-l"])

globa = "./images/FullIJCNN2013/"
names = []
namesTest = []
for i in range(43):
    if(i < 10):
        call(["rm", "-r", "./images/train/0" + str(i)])
        call(["rm", "-r", "./images/test/0" + str(i)])
        call(["mkdir", "./images/train/0" + str(i)])
        call(["mkdir", "./images/test/0" + str(i)])
        ims = glob.glob(globa + "0" + str(i) + "/*.ppm")
    else:
        call(["rm", "-r", "./images/train/" + str(i)])
        call(["rm", "-r", "./images/test/" + str(i)])
        call(["mkdir", "./images/train/" + str(i)])
        call(["mkdir", "./images/test/" + str(i)])
        ims = glob.glob(globa + str(i) + "/*.ppm")
    s = int(len(ims) * 0.8)
    names.extend(ims[0:s])
    namesTest.extend(ims[s:])


for i in names:
    l = cv2.imread(i)
    cv2.imwrite("./images/train/" + i.split("/")[-2] + "/" + i.split("/")[-1], l)

for i in namesTest:
    l = cv2.imread(i)
    cv2.imwrite("./images/test/" + i.split("/")[-2] + "/" + i.split("/")[-1], l)
