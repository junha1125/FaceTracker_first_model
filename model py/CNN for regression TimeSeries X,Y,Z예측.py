# 1.input set

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


train = img2array('C:/Users/admin/Desktop/0910-1/camera cap data') #4000 데이터

test = train[12000:]
train = train[0:12000]



# 2. sol set 만들기

import pandas
import matplotlib.pyplot as plt

total_file = 'C:/Users/admin/Desktop/0910-1/test.csv'

names = ['pitch','yaw','roll','x','y','z','state','frame','time_m','time']
df_total = pandas.read_csv(total_file, names=names)


#Total에서 y_train, y_test 나누기
a= list()
a.append(df_total.x[0:12000])
#a.append([df_total.x[0:12000],df_total.y[0:12000]])
a.append(df_total.x[12000:])
#a.append([df_total.x[12000:],df_total.y[12000:]])
y_train = np.array(a[0]).T
y_test = np.array(a[1]).T  # Transpose & array


y_train = y_train/315    # 1도 당 90.7 yaw transfom to angle
y_train = y_train[1:len(y_train)]

y_test = y_test/315
y_test = y_test[1:len(y_test)]


def data_diff(dataset):
    train_diff , train = [], []
    for i in range(len(dataset)-1):
        diff = (dataset[i+1] - dataset[i])
        
        train_diff.append(diff)                            #tuple 형 len= 2 [i+1]-[i], i+1 데이터 array반환
        train.append(dataset[i+1])


        
    return np.array(train_diff), np.array(train)              #자동으로 0번째 (쓰레기데이터) 무시됨


x_train_diff = data_diff(train)[0]
x_train = data_diff(train)[1]
x_train_diff = x_train_diff.astype('float32')

# noise 처리
#x_train_diff[x_train_diff<10] = 0
#x_train_diff[x_train_diff>220] = 0

x_train = x_train.astype('float32')
x_train_stack = np.stack((x_train,x_train_diff),axis=3)   # (data num , 128,128, 2)


x_test_diff = data_diff(test)[0]
x_test = data_diff(test)[1]
x_test_diff = x_test_diff.astype('float32')

# noise 처리
#x_test_diff[x_test_diff<10] = 0
#x_test_diff[x_test_diff>220] = 0

x_test = x_test.astype('float32')
x_test_stack = np.stack((x_test,x_test_diff),axis=3)   # (data num , 128,128, 2)

# normalize
x_train_stack = x_train_stack / 255
x_test_stack = x_test_stack / 255

print('x_train:',x_train_stack.shape)

print('x_test:' , x_test_stack.shape)

print('y_train:', y_train.shape)

print('y_test:',y_test.shape)


import keras.backend.tensorflow_backend as K
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from sklearn.preprocessing import MinMaxScaler

from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import optimizers

'''
## dataset 시각화
import matplotlib.pyplot as plt
import random



plt_row = 5
plt_col = 5

plt.rcParams["figure.figsize"] = (10,10)

    
f, axarr = plt.subplots(plt_row, plt_col)



for i in range(plt_row*plt_col):

    j=random.randrange(0,len(x_test)-1)
    sub_plt = axarr[i//plt_row, i%plt_col]
    sub_plt.axis('off')
    sub_plt.imshow(x_train[j])
    sub_plt.set_title('Real x %.1f\n Real y %.1f '
                      % (y_train[j][0],y_train[j][1])) 


plt.show()
'''

#model 구축
with K.tf.device('/gpu:0'):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=(135, 180,2)))  # output (128,128) 32개
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))  # output (64,64) 32개
    
    
    model.add(Conv2D(32, (3, 3 ), padding='same', activation='relu'))  # (64,64) 32개
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))  # output (32,32) 32개
    

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))  # (64,64) 32개
    
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))  # (64,64) 32개
    
    
    model.add(Flatten())

    model.add(Dense(128, init='normal' , activation='relu'))
    model.add(Dense(128, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(1,init='normal',activation=None,))  # (yaw,pitch,roll) 예측

    #Compile model
    learning_rate = 0.05
    epochs = 10
    decay_rate = learning_rate / epochs

    from keras.callbacks import EarlyStopping

    early_stopping = EarlyStopping()  # 개선여지 없을때 조기종료
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2 = 0.999, epsilon=None, decay=0.0, amsgrad=False)

    
    model.compile(loss= 'mse', optimizer='adam')
    hist_256 = model.fit(x_train_stack, y_train, validation_data=(x_test_stack,y_test),
                         batch_size=512, epochs=epochs,verbose=2 )



model.save('C:/Users/admin/Desktop/My_CNN_model_0827.h5')


print('My_CNN_model.h5저장완료')


# 5. 학습과정 보기 batch 256

import matplotlib.pyplot as plt

plt.plot(hist_256.history['loss'])
plt.plot(hist_256.history['val_loss'])
plt.ylim(0.0, 1000.0)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()




# 7. 모델 사용
yhat_test = model.predict(x_test_stack, batch_size= 128)

import matplotlib.pyplot as plt
import random
plt_row = 5
plt_col = 5

plt.rcParams["figure.figsize"] = (10, 10)

f, axarr = plt.subplots(plt_row, plt_col)

for i in range(plt_row * plt_col):

    j=random.randrange(0,len(x_test)-1) # 이미지 랜덤추출
    
    sub_plt = axarr[i // plt_row, i % plt_col]
    sub_plt.axis('off')
    sub_plt.imshow(x_test[j])
    sub_plt.set_title('R x %.1f R y %.1f \n P x %.1f P y %.1f  '
                      % (y_test[j][0],y_test[j][1], yhat_test[j][0],yhat_test[j][1]))


plt.show()



fig = plt.figure()
ax1=fig.add_subplot(2,1,1)
ax2=fig.add_subplot(2,1,2)

num = np.array(range(0,y_test.shape[0]))
num = num.reshape(num.shape[0],1)

ax1.plot(num,y_test[:,0], label ='real')
ax1.plot(num,yhat_test[:,0], label = 'predict')


ax2.plot(num,y_test[:,1], label='real')
ax2.plot(num,yhat_test[:,1], label='predict')
plt.xlabel('number')
plt.ylabel('x & y')
plt.title('difference real&predict')
plt.legend()
plt.show()


#more train

#from keras.models import load_model
#model = load_model('C:/Users/admin/Desktop/model 저장/My_CNN_model_0828.h5')




    
