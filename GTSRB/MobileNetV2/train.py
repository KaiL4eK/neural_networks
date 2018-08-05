import os
import sys
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split

from model import *
from data import *

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Reshape, Activation
from keras.models import Model
from keras.utils import to_categorical


def main(argv):
    parser = argparse.ArgumentParser()
    # Required arguments.
    parser.add_argument(
        "--classes",
        default=43,
        help="The number of classes of dataset.")
    # Optional arguments.
    parser.add_argument(
        "--size",
        default=48,
        help="The image size of train sample.")
    parser.add_argument(
        "--batch",
        default=32,
        help="The number of train samples per batch.")
    parser.add_argument(
        "--epochs",
        default=300,
        help="The number of train iterations.")
    parser.add_argument(
        "--weights",
        default=False,
        help="Fine tune with other weights.")
    parser.add_argument(
        "--tclasses",
        default=0,
        help="The number of classes of pre-trained model.")

    args = parser.parse_args()

    train(int(args.batch), int(args.epochs), int(args.classes), int(args.size), args.weights, int(args.tclasses))


def generate(batch, size):
    """Data generation and augmentation
    # Arguments
        batch: Integer, batch size.
        size: Integer, image size.
    # Returns
        train_generator: train set generator
        validation_generator: validation set generator
        count1: Integer, number of train set.
        count2: Integer, number of test set.
    """

    #  Using the data Augmentation in traning data
    ptrain = 'data/train'
    pval = 'data/validation'

    datagen1 = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen2 = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen1.flow_from_directory(
        ptrain,
        target_size=(size, size),
        batch_size=batch,
        class_mode='categorical')

    validation_generator = datagen2.flow_from_directory(
        pval,
        target_size=(size, size),
        batch_size=batch,
        class_mode='categorical')

    count1 = 0
    for root, dirs, files in os.walk(ptrain):
        for each in files:
            count1 += 1

    count2 = 0
    for root, dirs, files in os.walk(pval):
        for each in files:
            count2 += 1

    return train_generator, validation_generator, count1, count2




def train(batch, epochs, num_classes, size, weights, tclasses):
    """Train the model.
    # Arguments
        batch: Integer, The number of train samples per batch.
        epochs: Integer, The number of train iterations.
        num_classes, Integer, The number of classes of dataset.
        size: Integer, image size.
        weights, String, The pre_trained model weights.
        tclasses, Integer, The number of classes of pre-trained model.
    """

    imgs_data  = npy_data_load_images()
    lbls_data  = npy_data_load_labels()

    full_imgs_dict = {}

    for key in np.unique( lbls_data ):
        full_imgs_dict[key] = []

    max_class_num = np.max( lbls_data ) + 1

    fill_input_idx = 0
    imgs_train_input = []
    lbls_train_input = []

    imgs_test_input = []
    lbls_test_input = []


    for i in range(imgs_data.shape[0]):

        c_img = imgs_data[i].astype('float32', copy=False)
        c_img /= 255.

        # imgs_train_input[i] = c_img

        class_number = lbls_data[i]

        full_imgs_dict[class_number].append( c_img )

        # Convert to binary ( 0-42 - classes )
        # lbls_train_input[i] = to_categorical( class_number, num_classes=max_class_num )

        # print( class_number, lbls_train_input[i] )

    for key, value in full_imgs_dict.items():
        # print( key, len(value) )

        class_imgs = np.array(value)

        class_categ = to_categorical( key, num_classes=max_class_num )
        label_list = np.full( shape=(class_imgs.shape[0], max_class_num), fill_value=class_categ )

        img_train, img_test, lbl_train, lbl_test = train_test_split(class_imgs, label_list, test_size=0.1)

        for i, img in enumerate(img_train):
            imgs_train_input.append( img_train[i] )
            lbls_train_input.append( lbl_train[i] )

        for i, img in enumerate(img_test):
            imgs_test_input.append( img_test[i] )
            lbls_test_input.append( lbl_test[i] )

        # print( img_train.shape, lbl_train.shape )
        # print( img_test.shape, lbl_test.shape )
        # print( np.array(imgs_train_input).shape )

    imgs_train_input = np.array(imgs_train_input, dtype=np.float32)
    lbls_train_input = np.array(lbls_train_input, dtype=np.float32)
    imgs_test_input = np.array(imgs_test_input, dtype=np.float32)
    lbls_test_input = np.array(lbls_test_input, dtype=np.float32)

    print( imgs_train_input.shape, lbls_train_input.shape )
    print( imgs_test_input.shape, lbls_test_input.shape )

    model = MobileNetv2((size, size, 3), num_classes)

    opt = Adam()
    earlystop = EarlyStopping(monitor='val_acc', patience=30, verbose=0, mode='auto')
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    hist = model.fit( imgs_train_input, lbls_train_input, 
            batch_size=batch, 
            epochs=7000, 
            verbose=1, 
            shuffle=True, 
            validation_data=(imgs_test_input, lbls_test_input), 
            callbacks=[earlystop])

    # hist = model.fit_generator(
    #     train_generator,
    #     validation_data=validation_generator,
    #     steps_per_epoch=count1 // batch,
    #     validation_steps=count2 // batch,
    #     epochs=epochs,
    #     callbacks=[earlystop])

    if not os.path.exists('model'):
        os.makedirs('model')

    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/hist.csv', encoding='utf-8', index=False)
    model.save_weights('model/weights.h5')


if __name__ == '__main__':
    main(sys.argv)