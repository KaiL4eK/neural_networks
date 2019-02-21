import os
import pandas as pd
import models
import data
import generator as gen
import json
from keras.optimizers import Adam
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

    train_set, valid_set = data.create_training_instances(config['train']['train_folder'],
                                                          config['train']['train_masks'],
                                                          config['valid']['valid_folder'],
                                                          config['valid']['valid_masks'],
                                                          config['train']['cache_name'])

    train_generator = gen.BatchGenerator(
        instances           = train_set,
        batch_size          = config['train']['batch_size'],
        input_sz            = config['model']['input_shape'],
        shuffle             = True,
        jitter              = 0.3,
        norm                = data.normalize,
        downsample          = 2
    )

    valid_generator = gen.BatchGenerator(
        instances           = valid_set,
        batch_size          = config['train']['batch_size'],
        input_sz            = config['model']['input_shape'],
        norm                = data.normalize,
        infer               = True,
        downsample          = 2
    )

    early_stop = EarlyStopping(
        monitor     = 'val_loss',
        min_delta   = 0,
        patience    = 100,
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

    # Swapped as net input -> [H x W x C]
    net_input_shape = (config['model']['input_shape'][1],
                       config['model']['input_shape'][0],
                       3)

    train_model = models.create(
        base            = config['model']['base'],
        input_shape     = net_input_shape)

    if initial_weights:
        train_model.load_weights(initial_weights)

    model_render_file = 'images/{}.png'.format(config['model']['base'])
    if not os.path.isdir(os.path.dirname(model_render_file)):
        os.makedirs(os.path.dirname(model_render_file))

    plot_model(train_model, to_file=model_render_file, show_shapes=True)
    # print_summary(train_model)

    optimizer = Adam(lr=config['train']['learning_rate'], clipnorm=0.001)
    # optimizer = SGD(lr=config['train']['learning_rate'], clipnorm=0.001)

    train_model.compile(loss=models.result_loss, optimizer=optimizer,
                        metrics=[models.iou_loss, models.dice_coef_loss, models.pixelwise_crossentropy])

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

    hist = train_model.fit_generator(
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

    if not os.path.exists('model'):
        os.makedirs('model')

    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/hist.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    main()