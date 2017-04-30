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
from keras.models import load_model
import pandas as pd
import numpy as np

learning_rate = 0.01
img_width = 100
img_height = 100
nbr_train_samples = 3019
nbr_validation_samples = 758
nbr_epochs = 200
batch_size = 32
nbr_test_samples = 1000
nbr_augmentation = 1

test_data_dir = './input/test'


FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
model=load_model('./model/modelM_vgg16_2'  + '.h5')
#model= VGG16(weights='imagenet', include_top=False)
# this is the augmentation configuration we will use for training
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
test_aug=3;
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
        res=model.predict_generator(test_generator,13153)
    else:
        res += model.predict_generator(test_generator, 13153)
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
create_submission(res,test_generator.filenames)
