import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from matplotlib import pyplot
from sklearn.metrics import accuracy_score


#read all the images
img_shape = (32,48)
imgs = np.zeros(shape = (10000,) + img_shape + (3,), dtype = int)
for i in tqdm(range(10000)):
    img = cv2.imread('/content/drive/My Drive/task4_handout/food/food/' + str(i).zfill(5) + '.jpg')
    width  = img.shape[1]
    height = img.shape[0]
    start_width  = round(width  * 0.25)
    end_width    = round(width  * 0.75)
    start_height = round(height * 0.25)
    end_height   = round(height * 0.75)
    imgs[i,:,:,:] = cv2.resize(img[start_height:end_height, start_width:end_width, :], (img_shape[1],img_shape[0]))
    
train_triplets = np.loadtxt("/content/drive/My Drive/task4_handout/train_triplets.txt", dtype = int)
test_triplets  = np.loadtxt("/content/drive/My Drive/task4_handout/test_triplets.txt",  dtype = int)


num_trip = train_triplets.shape[0]
num_trip_test = test_triplets.shape[0]

X = np.zeros(shape = (num_trip*2,) + img_shape + (9,), dtype = 'float32')
test_X = np.zeros(shape = (num_trip_test,) + img_shape + (9,), dtype = 'float32')
test_X_inv = np.zeros(shape = (num_trip_test,) + img_shape + (9,), dtype = 'float32')

for i in tqdm(range(num_trip)):
    X[i, :, :, 0:3] = imgs[train_triplets[i,0], :, :, :]
    X[i, :, :, 3:6] = imgs[train_triplets[i,1], :, :, :]
    X[i, :, :, 6:9] = imgs[train_triplets[i,2], :, :, :]
    X[i + num_trip, :, :, 0:3] = imgs[train_triplets[i,0], :, :, :]
    X[i + num_trip, :, :, 3:6] = imgs[train_triplets[i,2], :, :, :]
    X[i + num_trip, :, :, 6:9] = imgs[train_triplets[i,1], :, :, :]
        
X /= 255

for i in tqdm(range(num_trip_test)):
    test_X[i, :, :, 0:3] = imgs[test_triplets[i,0], :, :, :]
    test_X[i, :, :, 3:6] = imgs[test_triplets[i,1], :, :, :]
    test_X[i, :, :, 6:9] = imgs[test_triplets[i,2], :, :, :]
    test_X_inv[i, :, :, 0:3] = imgs[test_triplets[i,0], :, :, :]
    test_X_inv[i, :, :, 3:6] = imgs[test_triplets[i,2], :, :, :]
    test_X_inv[i, :, :, 6:9] = imgs[test_triplets[i,1], :, :, :]
    
test_X /= 255
test_X_inv /= 255

del(imgs)
del(train_triplets)
del(test_triplets)
        
Y = np.append(np.ones((num_trip,), dtype = int), np.zeros((num_trip,), dtype = int))

train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size = 0.2)

model = Sequential()
model.add(Conv2D(32, (2,2), activation = 'relu', input_shape = img_shape+(9,), padding = 'same'))
model.add(MaxPooling2D((2,2), padding = 'same'))
model.add(Dropout(0.3))
model.add(Conv2D(64, (2,2), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D((2,2), padding = 'same'))
model.add(Dropout(0.3))
model.add(Conv2D(128, (2,2), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D((2,2), padding = 'same'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = "binary_crossentropy", optimizer = keras.optimizers.adam(learning_rate = 0.0001), metrics = ['accuracy'])

model_train = model.fit(train_X, train_Y, batch_size = 16, epochs = 20, verbose = 1, validation_data = (valid_X, valid_Y))
    
test_Y = (model.predict(test_X) > model.predict(test_X_inv)) * 1

np.savetxt('predict.txt', test_Y, fmt = '%d')

    
    




