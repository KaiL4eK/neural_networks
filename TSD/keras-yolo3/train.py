#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
from voc import parse_voc_annotation, split_by_objects
from yolo import create_model, dummy_loss
from generator import BatchGenerator
from utils.utils import normalize, evaluate, makedirs, init_session
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
from callbacks import CustomModelCheckpoint, CustomTensorBoard, MAP_evaluation
from keras.utils.layer_utils import print_summary

import keras
from keras.models import load_model


def create_training_instances(
        train_annot_folder,
        train_image_folder,
        train_cache,
        valid_annot_folder,
        valid_image_folder,
        valid_cache,
        labels,
):
    # parse annotations of the training set
    train_ints, train_labels = parse_voc_annotation(train_annot_folder, train_image_folder, train_cache, labels)

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(valid_annot_folder):
        valid_ints, valid_labels = parse_voc_annotation(valid_annot_folder, valid_image_folder, valid_cache, labels)
    else:
        print("valid_annot_folder not exists. Spliting the trainining set.")

        train_ints, valid_ints = split_by_objects(train_ints, train_labels, 0.2)

        # train_valid_split = int(0.8*len(train_ints))
        # np.random.seed(0)
        # np.random.shuffle(train_ints)
        # np.random.seed()

        # valid_ints = train_ints[train_valid_split:]
        # train_ints = train_ints[:train_valid_split]

    # compare the seen labels with the given labels in config.json
    if len(labels) > 0:
        overlap_labels = set(labels).intersection(set(train_labels.keys()))

        print('Seen labels: \t' + str(train_labels) + '\n')
        print('Given labels: \t' + str(labels))

        # return None, None, None if some given label is not in the dataset
        if len(overlap_labels) < len(labels):
            print('Some labels have no annotations! Please revise the list of labels in the config.json.')
            return None, None, None
    else:
        print('No labels are provided. Train on all seen labels.')
        print(train_labels)
        labels = train_labels.keys()

    max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])

    return train_ints, valid_ints, sorted(labels), max_box_per_image


def train(config, initial_weights):

    init_session(1.0)
    makedirs(os.path.dirname(config['train']['saved_weights_name']))
    makedirs(os.path.dirname(config['train']['cache_name']))

    ###############################
    #   Parse the annotations
    ###############################
    train_ints, valid_ints, labels, max_box_per_image = create_training_instances(
        config['train']['train_annot_folder'],
        config['train']['train_image_folder'],
        config['train']['cache_name'],
        config['valid']['valid_annot_folder'],
        config['valid']['valid_image_folder'],
        config['valid']['cache_name'],
        config['model']['labels']
    )
    print('\nTraining on: \t' + str(labels) + '\n')

    ###############################
    #   Create the generators
    ###############################
    train_generator = BatchGenerator(
        instances=train_ints,
        anchors=config['model']['anchors'],
        labels=labels,
        downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=max_box_per_image,
        batch_size=config['train']['batch_size'],
        min_net_size=config['model']['min_input_size'],
        max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=0.3,
        norm=normalize
    )

    valid_generator = BatchGenerator(
        instances=valid_ints,
        anchors=config['model']['anchors'],
        labels=labels,
        downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=max_box_per_image,
        batch_size=config['train']['batch_size'],
        norm=normalize,
        infer_sz=config['model']['infer_shape']
    )

    ###############################
    #   Create the model
    ###############################

    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    multi_gpu = len(config['train']['gpus'].split(','))

    freezing = True
    config['train']['warmup_epochs'] = 0

    warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times'] * len(train_generator))

    train_model, infer_model, _, freeze_num = create_model(
        nb_class=len(labels),
        anchors=config['model']['anchors'],
        max_box_per_image=max_box_per_image,
        max_input_size=config['model']['max_input_size'],
        batch_size=config['train']['batch_size'],
        warmup_batches=warmup_batches,
        ignore_thresh=config['train']['ignore_thresh'],
        multi_gpu=multi_gpu,
        grid_scales=config['train']['grid_scales'],
        obj_scale=config['train']['obj_scale'],
        noobj_scale=config['train']['noobj_scale'],
        xywh_scale=config['train']['xywh_scale'],
        class_scale=config['train']['class_scale'],
        base=config['model']['base']
    )

    from keras.utils.vis_utils import plot_model
    model_render_file = 'images/{}.png'.format(config['model']['base'])
    if not os.path.isdir(os.path.dirname(model_render_file)):
        os.makedirs(os.path.dirname(model_render_file))
    plot_model(infer_model, to_file=model_render_file, show_shapes=True)
    # print_summary(infer_model)

    # load the pretrained weight if exists, otherwise load the backend weight only
    if initial_weights and os.path.exists(initial_weights):
        print("\nLoading pretrained weights {}".format(initial_weights))
        train_model.load_weights(initial_weights, by_name=True, skip_mismatch=True)

        freezing = False

    ###############################
    #   Kick off the training
    ###############################
    tensorboard_logdir_idx = 0

    while True:
        tensorboard_logdir = "%s-%d" % (config['train']['tensorboard_dir'], tensorboard_logdir_idx)
        if os.path.exists(tensorboard_logdir):
            tensorboard_logdir_idx += 1
        else:
            break

    makedirs(tensorboard_logdir)

    print('Tensorboard dir: {}'.format(tensorboard_logdir))

    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=0.1,
        patience=3,
        mode='min',
        verbose=1
    )

    callbacks = [early_stop]

    if freezing and freeze_num > 0:
        print('Freezing %d layers of %d' % (freeze_num, len(infer_model.layers)))
        for l in range(freeze_num):
            infer_model.layers[l].trainable = False

        # optimizer = Adam(lr=1e-3, clipnorm=0.1)
        optimizer = Adam()
        train_model.compile(loss=dummy_loss, optimizer=optimizer)
        train_model.fit_generator(
            generator=train_generator,
            steps_per_epoch=len(train_generator) * config['train']['train_times'],

            validation_data=valid_generator,
            validation_steps=len(valid_generator) * config['valid']['valid_times'],

            epochs=config['train']['nb_epochs'] + config['train']['warmup_epochs'],
            verbose=2 if config['train']['debug'] else 1,
            callbacks=callbacks,
            workers=8,
            max_queue_size=100
        )

    # make a GPU version of infer_model for evaluation
    # if multi_gpu > 1:
    #     infer_model = load_model(config['train']['saved_weights_name'])

    print('Full training')

    for layer in infer_model.layers:
        layer.trainable = True

    root, ext = os.path.splitext(config['train']['saved_weights_name'])
    checkpoint_vloss = CustomModelCheckpoint(
        model_to_save=infer_model,
        filepath=root + '_ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}' + ext,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1
    )

    reduce_on_plateau = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=5,
        verbose=1,
        mode='min',
        min_delta=0.01,
        cooldown=0,
        min_lr=0
    )

    tensorboard_cb = TensorBoard(log_dir=tensorboard_logdir,
                                 histogram_freq=0,
                                 write_graph=True,
                                 write_images=False)

    map_evaluator_cb = MAP_evaluation(infer_model=infer_model,
                                      generator=valid_generator,
                                      save_best=True,
                                      save_name=root + '_best_mAP{mAP:.3f}' + ext,
                                      tensorboard=tensorboard_cb,
                                      iou_threshold=0.5,
                                      score_threshold=0.5)

    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=20,
        mode='min',
        verbose=1
    )

    from keras.optimizers import SGD

    callbacks = [checkpoint_vloss, tensorboard_cb, map_evaluator_cb]

    optimizer = Adam(lr=config['train']['learning_rate'], clipnorm=0.001)
    # optimizer = SGD(lr=config['train']['learning_rate'], clipnorm=0.001)

    train_model.compile(loss=dummy_loss, optimizer=optimizer)
    train_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator) * config['train']['train_times'],

        validation_data=valid_generator,
        validation_steps=len(valid_generator) * config['valid']['valid_times'],

        epochs=config['train']['nb_epochs'] + config['train']['warmup_epochs'],
        verbose=2 if config['train']['debug'] else 1,
        callbacks=callbacks,
        workers=8,
        max_queue_size=100
    )

    ###############################
    #   Run the evaluation
    ###############################
    # compute mAP for all the classes
    average_precisions = evaluate(infer_model, valid_generator)

    # print the score
    for label, average_precision in average_precisions.items():
        print(labels[label] + ': {:.4f}'.format(average_precision))
    print('Last mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-w', '--weights', help='path to pretrained model', default=None)
    args = argparser.parse_args()

    config_path = args.conf
    initial_weights = args.weights

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    train(config, initial_weights)
