#! /usr/bin/env python

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from _common import voc
import _common.callbacks as cbs
from _common import utils

import multiprocessing as mp
import os
import numpy as np
import json
import yolo
import tensorflow as tf
from my_generator import BatchGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import tensorflow.keras.optimizers as opt
# from keras_radam import RAdam
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session


import neptune
neptune.init('kail4ek/sandbox')


def parse_args():
    import argparse
    argparser = argparse.ArgumentParser(description='train YOLO model')
    argparser.add_argument(
        '-c', '--conf', help='path to configuration file'
    )
    argparser.add_argument(
        '-w', '--weights', help='path to pretrained model',
        default=None
    )
    argparser.add_argument(
        '-d', '--dry', help='dry run',
        action='store_true'
    )
    return argparser.parse_args()


def prepare_generators(config):
    utils.makedirs_4_file(config['train']['cache_name'])
    utils.makedirs_4_file(config['valid']['cache_name'])

    labels = config['model']['labels']

    # parse annotations of the training set
    train_ints, train_labels = voc.parse_voc_annotation(
        config['train']['annot_folder'],
        config['train']['image_folder'],
        config['train']['cache_name'],
        labels)

    # parse annotations of the validation set, if any, otherwise split the training set
    if config['valid']['annot_folder']:
        valid_ints, valid_labels = voc.parse_voc_annotation(
            config['valid']['annot_folder'],
            config['valid']['image_folder'],
            config['valid']['cache_name'],
            labels)
    else:
        from sklearn.model_selection import train_test_split

        print("valid_annot_folder not exists. Spliting the trainining set.")

        train_ints, valid_ints = train_test_split(train_ints,
                                                  random_state=37,
                                                  test_size=0.4)

        train_labels = voc.get_labels_dict(train_ints)
        valid_labels = voc.get_labels_dict(valid_ints)

        overlap_labels = set(train_labels.keys()).intersection(set(valid_labels.keys()))

        if len(overlap_labels) != len(train_labels.keys()) or \
           len(overlap_labels) != len(valid_labels.keys()):
            raise Exception('Invalid split of data: {} vs {}'.format(train_labels, valid_labels))

    print('After split: {} / {}'.format(len(train_ints), len(valid_ints)))

    # compare the seen labels with the given labels in config.json
    if len(labels) > 0:
        overlap_labels = set(labels).intersection(set(train_labels.keys()))

        print('Seen labels: \t' + str(train_labels) + '\n')
        print('Given labels: \t' + str(labels))

        # return None, None, None if some given label is not in the dataset
        if len(overlap_labels) != len(labels):
            raise Exception('Some labels have no annotations! Please revise the list of labels in the config.json.')
    else:
        print('No labels are provided. Train on all seen labels.')
        print(train_labels)
        print(valid_labels)
        labels = list(train_labels.keys())

    max_box_per_image = max([len(inst['object'])
                             for inst in (train_ints + valid_ints)])

    # Dirty hack
    # train_ints, train_labels = voc.replace_all_labels_2_one(train_ints, 'object')
    # valid_ints, valid_labels = voc.replace_all_labels_2_one(valid_ints, 'object')
    # labels = list(train_labels.keys())

    print('\n')
    print('Training on: \t{}'.format(train_labels))
    print('Validating on: \t{}'.format(valid_labels))
    print('Labels: \t{}'.format(labels))
    print('Samples: {} / {}\t'.format(len(train_ints), len(valid_ints)))
    print('\n')

    ###############################
    #   Create the generators
    ###############################
    train_generator = BatchGenerator(
        instances=train_ints,
        anchors=config['model']['anchors'],
        anchors_per_output=config['model']['anchors_per_output'],
        labels=labels,
        # ratio between network input's size and network output's size, 32 for YOLOv3
        downsample=config['model']['downsample'],
        max_box_per_image=max_box_per_image,
        batch_size=config['train']['batch_size'],
        min_net_size=config['train']['min_input_size'],
        max_net_size=config['train']['max_input_size'],
        shuffle=True,
        aug_params=config['train'].get('augmentation', None),
        norm=utils.image_normalize,
        tile_count=config['model']['tiles']
    )

    valid_generator = BatchGenerator(
        instances=valid_ints,
        anchors=config['model']['anchors'],
        anchors_per_output=config['model']['anchors_per_output'],
        labels=labels,
        # ratio between network input's size and network output's size, 32 for YOLOv3
        downsample=config['model']['downsample'],
        max_box_per_image=max_box_per_image,
        batch_size=config['train']['batch_size'],
        norm=utils.image_normalize,
        infer_sz=config['model']['infer_shape'],
        tile_count=config['model']['tiles']
    )

    # Updated based on data parsing
    config['train']['mbpi'] = max_box_per_image
    config['model']['labels'] = labels

    return train_generator, valid_generator


def prepare_model(config, initial_weights):

    if initial_weights and os.path.exists(initial_weights):
        freezing = False

    yolo_model = yolo.YOLO_Model(
        config['model'],
        config['train']
    )

    # load the pretrained weight if exists, otherwise load the backend weight only
    if initial_weights:
        yolo_model.load_weights(initial_weights)

    return yolo_model


def train_freezed(config, train_model, train_generator, valid_generator):
    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=0.1,
        patience=3,
        mode='min',
        verbose=1
    )

    callbacks = [early_stop]

    if freezing:
        optimizer = Adam(lr=1e-3)
        train_model.compile(loss=yolo.dummy_loss, optimizer=optimizer)
        train_model.fit_generator(
            generator=train_generator,
            steps_per_epoch=len(train_generator) *
            config['train']['train_times'],

            validation_data=valid_generator,
            validation_steps=len(valid_generator) *
            config['valid']['valid_times'],

            epochs=config['train']['nb_epochs'] +
            config['train']['warmup_epochs'],
            verbose=2 if config['train']['debug'] else 1,
            callbacks=callbacks,
            workers=8,
            max_queue_size=100
        )

    # make a GPU version of infer_model for evaluation
    # if multi_gpu > 1:
    #     infer_model = load_model(config['train']['saved_weights_name'])

    utils.unfreeze_model(infer_model)


def start_train(
    config,
    config_path,
    yolo_model: yolo.YOLO_Model,
    train_generator,
    valid_generator,
    dry_mode: bool
):
    print('Full training')

    ###############################
    #   Optimizers
    ###############################

    optimizers = {
        'sgd': opt.SGD(lr=config['train']['learning_rate']),
        'adam': opt.Adam(lr=config['train']['learning_rate']),
        'adamax': opt.Adamax(lr=config['train']['learning_rate']),
        'nadam': opt.Nadam(lr=config['train']['learning_rate']),
        'rmsprop': opt.RMSprop(lr=config['train']['learning_rate']),
        # 'Radam': RAdam(lr=config['train']['learning_rate'], warmup_proportion=0.1, min_lr=1e-5)
    }

    optimizer = optimizers[config['train']['optimizer'].lower()]

    if config['train']['clipnorm'] > 0:
        optimizer.clipnorm = config['train']['clipnorm']

    if config['train'].get('lr_decay', 0) > 0:
        optimizer.decay = config['train']['lr_decay']

    if config['train']['optimizer'] == 'Nadam':
        # Just to set field
        optimizer.decay = 0.0

    ###############################
    #   Callbacks
    ###############################

    checkpoint_name = utils.get_checkpoint_name(config)
    utils.makedirs_4_file(checkpoint_name)

    checkpoint_vloss = cbs.CustomModelCheckpoint(
        model_to_save=yolo_model.infer_model,
        filepath=checkpoint_name,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1
    )

    # tensorboard_logdir = utils.get_tensorboard_name(config)
    # utils.makedirs(tensorboard_logdir)
    # print('Tensorboard dir: {}'.format(tensorboard_logdir))

    # tensorboard_cb = TensorBoard(
    #     log_dir=tensorboard_logdir,
    #     histogram_freq=0,
    #     write_graph=False
    # )

    mAP_checkpoint_name = utils.get_mAP_checkpoint_name(config)
    mAP_checkpoint_static_name = utils.get_mAP_checkpoint_static_name(config)
    utils.makedirs_4_file(mAP_checkpoint_name)
    map_evaluator_cb = cbs.MAP_evaluation(
        model=yolo_model,
        generator=valid_generator,
        save_best=True,
        save_name=mAP_checkpoint_name,
        save_static_name=mAP_checkpoint_static_name,
        # tensorboard=tensorboard_cb,
        neptune=neptune if not dry_mode else None
    )

    reduce_on_plateau = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.4,
        patience=20,
        verbose=1,
        mode='min',
        min_delta=0,
        cooldown=10,
        min_lr=1e-8
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=80,
        mode='min',
        verbose=1
    )

    neptune_mon = cbs.NeptuneMonitor(
        monitoring=['loss', 'val_loss'],
        neptune=neptune
    )

    # logger_cb = cbs.CustomLogger(
    #     config=config,
    #     tensorboard=tensorboard_cb
    # )

    # fps_logger = cbs.FPSLogger(
    #     infer_model=yolo_model.infer_model,
    #     generator=valid_generator,
    #     infer_sz=config['model']['infer_shape'],
    #     tensorboard=tensorboard_cb
    # )

    callbacks = [
        # tensorboard_cb,
        map_evaluator_cb,
        # early_stop,
        reduce_on_plateau,
    ]

    ###############################
    #   Prepare fit
    ###############################

    if not dry_mode:
        callbacks.append(neptune_mon)

        with open('config.json', 'w') as f:
            json.dump(config, f, indent=4)

        sources_to_upload = [
            'yolo.py',
            '_common/backend.py',
            'config.json'
        ]

        params = {
            'base_params': str(config['model']['base_params']),
            'infer_size': "H{}xW{}".format(*config['model']['infer_shape']),
            'anchors_per_output': config['model']['anchors_per_output'],
            'anchors': str(config['model']['anchors'])
        }
        
        tags = [
            config['model']['base']
        ]

        logger.info('Tags: {}'.format(tags))
        
        neptune.create_experiment(
            name=utils.get_neptune_name(config),
            upload_stdout=False,
            upload_source_files=sources_to_upload,
            params=params,
            tags=tags
        )
    else:
        config['train']['nb_epochs'] = 10

    yolo_model.train_model.compile(loss=yolo.dummy_loss, optimizer=optimizer)
    yolo_model.train_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator) * config['train']['train_times'],

        validation_data=valid_generator,
        validation_steps=len(valid_generator) * config['valid']['valid_times'],

        epochs=config['train']['nb_epochs'],
        verbose=1,
        callbacks=callbacks,
        workers=mp.cpu_count(),
        max_queue_size=100,
        use_multiprocessing=False
    )

    if not dry_mode:
        neptune.send_artifact(mAP_checkpoint_static_name)
        neptune.send_artifact('config.json')


if __name__ == '__main__':
    args = parse_args()

    utils.init_session()

    config_path = args.conf
    initial_weights = args.weights
    dry_mode = args.dry

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    train_generator, valid_generator = prepare_generators(config)

    yolo_model = prepare_model(config, initial_weights)

    # if freezing:
    # train_freezed(config, train_model, train_generator, valid_generator)

    start_train(config, config_path, yolo_model, train_generator, valid_generator, dry_mode)

    if not dry_mode:
        neptune.stop()

    clear_session()
