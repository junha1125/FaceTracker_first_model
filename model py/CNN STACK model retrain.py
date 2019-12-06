# image to array transform for training
import os
import numpy as np
from PIL import Image

def img2array(imagepath):
    img_arr_list = list()
    for (path, dirname, files) in os.walk('%s' % imagepath):
        for f in files:
            os.chdir('%s' % imagepath)
            img = Image.open(f)
            arr = np.array(img)
            img_arr_list.append(arr)  # list화
    return img_arr_list


train1 = img2array('C:/Users/admin/Desktop/yolo_py/0925kim/camera cap data')

#train2 = img2array('C:/Users/admin/Desktop/yolo_py/0923kwak/camera cap data')

train = train1
x_train = img_train = np.array(train)  # nd array for test

x_train = x_train.astype('float32')

import pandas
import matplotlib.pyplot as plt

train_file1 = 'C:/Users/admin/Desktop/yolo_py/0925kim/angle_data_suit.csv'

#train_file2 = 'C:/Users/admin/Desktop/yolo_py/0923kwak/angle_data_suit.csv'
names = ['num', 'pitch', 'yaw', 'roll']


df_train1 = pandas.read_csv(train_file1, names=names)
#df_train2 = pandas.read_csv(train_file2, names=names)

df_train = df_train1
#df_train = df_train1.append(df_train2)

a = list()
#a.append(df_train.pitch)
a.append(df_train.yaw)



y_train = np.array(a).T


y_train = y_train/91 # 1도 당 90.7 yaw transfom to angle


y_train = y_train[1:len(y_train)]

y_train_ori = y_train

import cv2

def data_diff(dataset):
    train_diff, train = [], []
    for i in range(len(dataset)-1):
        diff= cv2.subtract(dataset[i+1],dataset[i])
        train_diff.append(diff)
        train.append(dataset[i+1])                            #tuple 형 len= 2 [i+1]-[i], i+1 데이터 array반환
    return np.array(train_diff), np.array(train)              #자동으로 0번째 (쓰레기데이터) 무시됨



x_train_diff = data_diff(x_train)[0]
x_train = data_diff(x_train)[1]

x_train_diff = x_train_diff.astype('float32')
x_train = x_train.astype('float32')

x_train_stack = np.stack((x_train,x_train_diff),axis=3)   #원래 data + diff -1까지 합침



# normalize
x_train_stack = x_train_stack / 255

#x_train_stack_test = x_train_stack[21000:]
#x_train_stack = x_train_stack[:21000]


#y_train_test = y_train[21000:]
#y_train = y_train[:21000]

print(x_train_stack.shape)
print(y_train.shape)
#print(x_train_stack_test.shape)
#print(y_train_test.shape)

print('x_train_shape:',x_train.shape)
print('x_train_diff_shape:',x_train_diff.shape)



from keras.models import load_model

#STACK MODEL

stack_model = load_model('C:/Users/admin/Desktop/stack_model_yolo_v5.h5')

stack_model.fit(x_train_stack,y_train, validation_split = 0.15,shuffle=True,
          batch_size=512, epochs=1000, verbose=2)

stack_model.save('C:/Users/admin/Desktop/stack_model_yolo_v6.h5')
'''
#CNN MODEL

x_train = x_train.reshape(x_train.shape[0],128,128,1)
x_train = x_train/255

CNN_model = load_model('C:/Users/admin/Desktop/model_yolo_CNN_v4.h5')

CNN_model.fit(x_train,y_train_ori, validation_split = 0.15,shuffle=True,
          batch_size=512, epochs=500, verbose=2)

CNN_model.save('C:/Users/admin/Desktop/model_yolo_CNN_v5.h5')


#DIFF MODEL

x_train_diff = x_train_diff.reshape(x_train.shape[0],128,128,1)
x_train_diff = x_train_diff/255

diff_model = load_model('C:/Users/admin/Desktop/model_diff_yolo_CNN_v4.h5')

diff_model.fit(x_train_diff,y_train_ori, validation_split = 0.15,shuffle=True,
          batch_size=512, epochs=500, verbose=2)

diff_model.save('C:/Users/admin/Desktop/model_diff_yolo_CNN_v5.h5')
'''
# MODEL USE
y_hat_stack = stack_model.predict(x_train_stack, batch_size=128)

#y_hat_CNN = CNN_model.predict(x_train, batch_size=128)

#y_hat_diff = diff_model.predict(x_train_diff, batch_size=128)

# STACK model test 시각화
num = np.array(range(0,y_train.shape[0]))
num = num.reshape(num.shape[0],1)
plt.plot(num,y_train, label ='real')
plt.plot(num,y_hat_stack, label = 'predict')

plt.xlabel('number')
plt.ylabel('yaw')
plt.title('difference real&predict')
plt.legend()
plt.show()

# CNN model test 시각화
num = np.array(range(0,y_train_ori.shape[0]))
num = num.reshape(num.shape[0],1)
plt.plot(num,y_train_ori, label ='real')
plt.plot(num,y_hat_CNN, label = 'predict')

plt.xlabel('number')
plt.ylabel('yaw')
plt.title('difference real&predict')
plt.legend()
plt.show()

# diff model test 시각화
num = np.array(range(0,y_train_ori.shape[0]))
num = num.reshape(num.shape[0],1)
plt.plot(num,y_train_ori, label ='real')
plt.plot(num,y_hat_diff, label = 'predict')

plt.xlabel('number')
plt.ylabel('yaw')
plt.title('difference real&predict')
plt.legend()
plt.show()
