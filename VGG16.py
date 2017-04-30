import datetime
from keras.applications import VGG16
from keras.applications.inception_v3 import InceptionV3
import os
from keras.layers import Flatten, Dense, AveragePooling2D, ZeroPadding2D, Convolution2D, MaxPooling2D, \
    BatchNormalization, Dropout
from keras.models import Model, Sequential
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import pandas as pd

import numpy as np

learning_rate = 0.0001
img_width = 100
img_height = 100
nbr_train_samples = 3019
nbr_validation_samples = 758
nbr_epochs = 200
batch_size = 32
nbr_test_samples = 1000
nbr_augmentation = 1

train_data_dir = './input/new_train/train_split'
val_data_dir = './input/new_train/val_split'

FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
def create_model(data_shape,batch_size = 32,nb_classes = 10,nb_epoch = 200,data_augmentation = True):
    model = Sequential()
    vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3, 1, 1))
    model.add(ZeroPadding2D((1, 1), input_shape=data_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))





    model.load_weights('/home/dante0shy/.keras/models/vgg16_weights_th_dim_ordering_th_kernels_notop.h5')

    #model = VGG16(weights='imagenet', include_top=False)
    #model.
    for layer in model.layers:
        layer.trainable = False
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))#512
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))
    callbacks = [EarlyStopping(monitor='val_l'
                                       'oss', patience=1, verbose=0)]
    sgd = SGD(lr=learning_rate, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer="adadelta", metrics=["accuracy"])#
    return model

model=create_model((3,img_width,img_height))

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

# this is the augmentation configuration we will use for validation:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        classes = FishNames,
        class_mode = 'categorical')

validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        classes = FishNames,
        class_mode = 'categorical')
#print validation_generator.classes
##print train_generator.image_shape
#print train_generator.classes
#print train_generator.directory
print 'Start training'
model.fit_generator(
        train_generator,
        samples_per_epoch = nbr_train_samples,
        nb_epoch = nbr_epochs,
        validation_data = validation_generator,
        nb_val_samples = nbr_validation_samples,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0,mode='auto')])
model.save_weights('./model/weightM_vgg16_2' + '.h5')
model.save('./model/modelM_vgg16_2'  + '.h5')
test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for validation:
# only rescaling


#print validation_generator.classes
##print train_generator.image_shape
#print train_generator.classes
#print train_generator.directory
print 'Start testing'
test_data_dir = './input/test'
test_aug=5;
for i in range(test_aug):
    random_seed = np.random.random_integers(0, 100000)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size = (img_width, img_height),
        batch_size=32,
        shuffle=False,
        seed=random_seed,
        classes=None,
        class_mode=None)
    if i==0:
        res=model.predict_generator(test_generator,1000)
    else:
        res += model.predict_generator(test_generator, 1000)
res /= test_aug


Name_fish = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

def create_submission(predictions, test_id):
    print('building submission')

    id=[]
    for i in test_id:
        id.append(i[10:])
    result1 = pd.DataFrame(predictions, columns=Name_fish)
    result1.loc[:, 'image'] = pd.Series(id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = './output/submission_'  + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)
    #f_submit=open(sub_file,'w')
    #f_submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
    #for i, image_name in enumerate(test_generator.filenames):
    #    pred = [ p for p in predictions[i, :]]
    #    if i % 100 == 0:
    #        print(i, '/', 1000)
    #    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

    #f_submit.close()

create_submission(res,test_generator.filenames)