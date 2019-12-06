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


total1 = img2array('C:/Users/admin/Desktop/0828/camera cap data') #23550 데이터
total2 = img2array('C:/Users/admin/Desktop/0823/camera cap data') #18950
total3 = img2array('C:/Users/admin/Desktop/0821/camera cap data') #14014 = 56514

total =  total1 + total2 + total3 

train = total[0:53000]
test = total[53000:]




#test = img2array('C:/Users/admin/Desktop/test   data원본/0820/camera cap data')
'''
x_test = img_test = np.array(test)  # nd array for train
x_train = img_train = np.array(train)  # nd array for test

x_test = x_test.astype('float32')
x_train = x_train.astype('float32')


# normalize
x_train = x_train / 255
x_test = x_test / 255
'''
# 2. sol set 만들기

import pandas
import matplotlib.pyplot as plt

total_file1 = 'C:/Users/admin/Desktop/0828/angle_data_suit.csv'
total_file2 = 'C:/Users/admin/Desktop/0823/angle_data_suit.csv'
total_file3 = 'C:/Users/admin/Desktop/0821/angle_data_suit.csv'

#test_file = 'C:/Users/admin/Desktop/test   data원본/0820/angle_data_suit.csv'

names = ['num', 'pitch', 'yaw', 'roll']

df_total1 = pandas.read_csv(total_file1, names=names)
df_total2 = pandas.read_csv(total_file2, names=names)
df_total3 = pandas.read_csv(total_file3, names=names)

df_total = df_total1.append(df_total2).append(df_total3)
#df_test = pandas.read_csv(test_file, names=names)


#Total에서 y_train, y_test 나누기
a= list()
a.append(df_total.yaw[0:53000])
a.append(df_total.yaw[53000:])

y_train = np.array(a[0]).T
y_test = np.array(a[1]).T  # Transpose & array


y_train = y_train/91 # 1도 당 90.7 yaw transfom to angle
y_train = y_train[1:len(y_train)]

y_test = y_test/91
y_test = y_test[1:len(y_test)]




################################################# [t+1] - [t] 

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
x_train_diff[x_train_diff<10] = 0
x_train_diff[x_train_diff>220] = 0

x_train = x_train.astype('float32')
x_train_stack = np.stack((x_train,x_train_diff),axis=3)   # (data num , 128,128, 2)


x_test_diff = data_diff(test)[0]
x_test = data_diff(test)[1]
x_test_diff = x_test_diff.astype('float32')
# noise 처리
x_test_diff[x_test_diff<10] = 0
x_test_diff[x_test_diff>220] = 0

x_test = x_test.astype('float32')
x_test_stack = np.stack((x_test,x_test_diff),axis=3)   # (data num , 128,128, 2)






'''
################################################# [t+1] - [t] [t+2] - [t+1]

def data_diff(dataset):
    train_diff_1, train_diff_2 , train = [], [], []

    
    for i in range(len(dataset)-2):
        diff_1 = (dataset[i+2]-dataset[i+1])
        diff_2 = (dataset[i+2]-dataset[i])/2


        train_diff_1.append(diff_1)
        train_diff_2.append(diff_2)
        
        train.append(dataset[i+2])                            #tuple 형 len= 2 [i+1]-[i], i+1 데이터 array반환


        
    return np.array(train_diff_1),np.array(train_diff_2), np.array(train)              #자동으로 0번째 (쓰레기데이터) 무시됨

x_train_diff_1 = data_diff(train)[0]
x_train_diff_2 = data_diff(train)[1]
x_train = data_diff(train)[2]

x_train_diff_1 = x_train_diff_1.astype('float32')
x_train_diff_2 = x_train_diff_2.astype('float32')
x_train = x_train.astype('float32')






x_train_stack = np.stack((x_train,x_train_diff_1,x_train_diff_2),axis=3)   #원래 data + diff -1까지 합침




x_test_diff_1 = data_diff(test)[0]
x_test_diff_2 = data_diff(test)[1]
x_test = data_diff(test)[2]

x_test_diff_1 = x_test_diff_1.astype('float32')
x_test_diff_2 = x_test_diff_2.astype('float32')
x_test = x_test.astype('float32')

x_test_stack = np.stack((x_test,x_test_diff_1, x_test_diff_2),axis=3)   # (data num , 128,128, 2)





'''





'''

def data_past(dataset):
    train_past , train = [], []
    for i in range(len(dataset)-1):
        past = (dataset[i])
        
        train_past.append(past)
        train.append(dataset[i+1])                            #tuple 형 len= 2 [i+1]-[i], i+1 데이터 array반환


        
    return np.array(train_past), np.array(train)              #자동으로 0번째 (쓰레기데이터) 무시됨







x_train_past = data_past(train)[0]
x_train = data_past(train)[1]

x_train_past = x_train_past.astype('float32')
x_train = x_train.astype('float32')



x_train_stack = np.stack((x_train,x_train_past),axis=3)   #원래 data + diff -1까지 합침




x_test_past = data_past(test)[0]
x_test = data_past(test)[1]
x_test_past = x_test_past.astype('float32')
x_test = x_test.astype('float32')

x_test_stack = np.stack((x_test,x_test_past),axis=3)   # (data num , 128,128, 2)
'''

'''
x_train = x_train.reshape(x_train.shape[0],128,128,1).astype('float32')
x_test = x_test.reshape(x_test.shape[0],128,128,1).astype('float32')

'''




# normalize
x_train_stack = x_train_stack / 255
x_test_stack = x_test_stack / 255


# val train split


x_train_stack_val = x_train_stack[52000:]
x_train_stack = x_train_stack[:52000]


y_train_val = y_train[52000:]
y_train = y_train[:52000]

print('x_train:',x_train_stack.shape)
print('x_train_val:',x_train_stack_val.shape)
print('x_test:' , x_test_stack.shape)

print('y_train:', y_train.shape)
print('y_train_val:',y_train_val.shape)
print('y_test:',y_test.shape)


#model 



import keras.backend.tensorflow_backend as K
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from sklearn.preprocessing import MinMaxScaler

from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import optimizers





#x_train = x_train.reshape(x_train.shape[0],128,128,1).astype('float32')
#x_train_diff = x_train_diff.reshape(x_train_diff.shape[0],128,128,1).astype('float32')
#x_train_diff_val = x_train_diff_val.reshape(x_train_diff_val.shape[0],128,128,1).astype('float32')

#x_test = x_test.reshape(x_test.shape[0],128,128,1).astype('float32')
#x_test_diff = x_test_diff.reshape(x_test_diff.shape[0],128,128,1).astype('float32')


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
    sub_plt.imshow(x_train[j].reshape(96,128))
    sub_plt.set_title('Real yaw %.1f '
                      % (y_train[j])) 


plt.show()

'''


#model 구축
with K.tf.device('/gpu:0'):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=(96, 128,2)))  # output (128,128) 32개
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
    learning_rate = 0.01
    epochs = 100
    decay_rate = learning_rate / epochs

    from keras.callbacks import EarlyStopping

    early_stopping = EarlyStopping()  # 개선여지 없을때 조기종료
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2 = 0.999, epsilon=None, decay=0.0, amsgrad=False)

    
    model.compile(loss= 'mse', optimizer='adam')
    hist_256 = model.fit(x_train_stack, y_train, validation_data=(x_train_stack_val,y_train_val),
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

plt_row = 5
plt_col = 5

plt.rcParams["figure.figsize"] = (10, 10)

f, axarr = plt.subplots(plt_row, plt_col)

for i in range(plt_row * plt_col):

    j=random.randrange(0,len(x_test)-1) # 이미지 랜덤추출
    
    sub_plt = axarr[i // plt_row, i % plt_col]
    sub_plt.axis('off')
    sub_plt.imshow(x_test[j].reshape(96,128))
    sub_plt.set_title('R yaw %.1f \n P yaw %.1f  '
                      % (y_test[j], yhat_test[j]))


plt.show()

# y_test, y_prediction 시각화
fig = plt.figure()
ax1=fig.add_subplot(2,1,1)
#ax2=fig.add_subplot(2,1,2)

num = np.array(range(0,y_test.shape[0]))
num = num.reshape(num.shape[0],1)

ax1.plot(num,y_test[:], label ='real')
ax1.plot(num,yhat_test[:], label = 'predict')


#ax2.plot(num,y_test[:,1], label='real')
#ax2.plot(num,yhat_test[:,1], label='predict')
plt.xlabel('number')
plt.ylabel('pitch')
plt.title('difference real&predict')
plt.legend()
plt.show()
'''

#more train

from keras.models import load_model
model = load_model('C:/Users/admin/Desktop/model 저장/My_CNN_model_0828.h5')
model.fit(x_train_stack,y_train, epochs= 100, batch_size= 512,
          validation_data = (x_train_stack_val,y_train_val),
          verbose=2)



model.save('C:/Users/admin/Desktop/My_CNN_model_0828_1.h5')
'''
