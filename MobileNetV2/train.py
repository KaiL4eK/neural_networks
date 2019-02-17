import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import models
import data
import generator as gen
import json
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from keras.utils.layer_utils import print_summary

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from callbacks import CustomModelCheckpoint

import argparse
argparser = argparse.ArgumentParser(description='train and evaluate YOLOv3 model on any dataset')
argparser.add_argument('-c', '--conf', help='path to configuration file')
argparser.add_argument('-w', '--weights', help='path to trained model', default=None)

args = argparser.parse_args()


def main():
    config_path = args.conf
    initial_weights = args.weights

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    train_set, valid_set, classes = data.create_training_instances(config['train']['train_folder'],
                                                                   None,
                                                                   config['train']['cache_name'])

    num_classes = len(classes)
    print('Readed {} classes: {}'.format(num_classes, classes))

    train_generator = gen.BatchGenerator(
        instances           = train_set,
        labels              = classes,
        batch_size          = config['train']['batch_size'],
        input_sz            = config['model']['input_side_sz'],
        shuffle             = True,
        jitter              = 0.3,
        norm                = data.normalize
    )

    valid_generator = gen.BatchGenerator(
        instances           = valid_set,
        labels              = classes,
        batch_size          = config['train']['batch_size'],
        input_sz            = config['model']['input_side_sz'],
        norm                = data.normalize,
        infer               = True
    )

    early_stop = EarlyStopping(
        monitor     = 'val_loss',
        min_delta   = 0.1,
        patience    = 3,
        mode        = 'min',
        verbose     = 1
    )

    reduce_on_plateau = ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.5,
        patience = 5,
        verbose  = 1,
        mode     = 'min',
        min_delta= 0.01,
        cooldown = 0,
        min_lr   = 0
    )

    net_input_shape = (config['model']['input_side_sz'],
                       config['model']['input_side_sz'],
                       3)

    train_model = models.create(
        base            = config['model']['base'],
        num_classes     = num_classes,
        input_shape     = net_input_shape)

    print_summary(train_model)
    plot_model(train_model, to_file='images/MobileNetv2.png', show_shapes=True)

    optimizer = Adam(lr=config['train']['learning_rate'], clipnorm=0.001)
    # optimizer = SGD(lr=config['train']['learning_rate'], clipnorm=0.001)

    train_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    chk_name = config['train']['saved_weights_name']
    chk_root, chk_ext = os.path.splitext(chk_name)
    checkpoint_vloss = CustomModelCheckpoint(
        model_to_save   = train_model,
        filepath        = chk_root+'_ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}'+chk_ext,
        monitor         = 'val_loss',
        verbose         = 1,
        save_best_only  = True,
        mode            = 'min',
        period          = 1
    )

    if chk_name:
        if not os.path.isdir(os.path.dirname(chk_name)):
            os.makedirs(os.path.dirname(chk_name))

    callbacks = [early_stop, reduce_on_plateau, checkpoint_vloss]

    train_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator) * config['train']['train_times'],

        validation_data=valid_generator,
        validation_steps=len(valid_generator) * config['valid']['valid_times'],

        epochs=config['train']['nb_epochs'],
        verbose=2 if config['train']['debug'] else 1,
        callbacks=callbacks,
        workers=os.cpu_count(),
        max_queue_size=100
    )

    exit(1)


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
            batch_size=config['train']['batch_size'],
            epochs=config['train']['nb_epochs'],
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
    main()