import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback

import _common.utils as c_ut

class CustomTensorBoard(TensorBoard):
    """ to log the loss after each batch
    """

    def __init__(self, log_every=1, **kwargs):
        super(CustomTensorBoard, self).__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0

    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        if self.counter % self.log_every == 0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()

        super(CustomTensorBoard, self).on_batch_end(batch, logs)


class CustomModelCheckpoint(ModelCheckpoint):
    """ to save the template model, not the multi-GPU model
    """

    def __init__(self, model_to_save, **kwargs):
        super(CustomModelCheckpoint, self).__init__(**kwargs)
        self.model_to_save = model_to_save

    def on_epoch_end(self, epoch, logs={}):

        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model_to_save.save_weights(
                                filepath, overwrite=True)
                        else:
                            self.model_to_save.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' %
                          (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model_to_save.save_weights(filepath, overwrite=True)
                else:
                    self.model_to_save.save(filepath, overwrite=True)

        super(CustomModelCheckpoint, self).on_batch_end(epoch, logs)


import tensorflow.keras.backend as K
import time

class FPSLogger(Callback):
    def __init__(self,
                 infer_model,
                 generator,
                 infer_sz,
                 tensorboard=None):

        self.infer_model = infer_model
        self.generator = generator
        self.tensorboard = tensorboard
        self.infer_sz = infer_sz

        self.fps_test_imgs_cnt = 1
        self.batch_input = np.zeros((self.fps_test_imgs_cnt, self.infer_sz[0], self.infer_sz[1], 3))
        # for i in range(self.fps_test_imgs_cnt):
            # self.batch_input[i] = preprocess_input(self.generator.load_image(i), self.infer_sz[0], self.infer_sz[1])

        if not isinstance(self.tensorboard, TensorBoard) and self.tensorboard is not None:
            raise ValueError(
                "Tensorboard object must be a instance from keras.callbacks.TensorBoard")

    # def on_train_begin(self, logs=None):
        # if self.tensorboard is not None and self.tensorboard.writer is not None:
        #     print('Pretraining FPS estimation')
        #     for i in range(100):
        #         self.infer_model.predict_on_batch(self.batch_input)

        #     start_time = time.time()
        #     self.infer_model.predict_on_batch(self.batch_input)
        #     result_time = (time.time() - start_time) / self.fps_test_imgs_cnt

        #     print('Pretraining FPS estimation done')

        #     hyperparameters = [tf.convert_to_tensor(['inf_time', str(result_time)])]
        #     summary = tf.summary.text('logs=)', tf.stack(hyperparameters))
        #     self.tensorboard.writer.add_summary(K.eval(summary))

    # def on_epoch_end(self, epoch, logs=None):

    #     if self.tensorboard is not None and self.tensorboard.writer is not None:
    #         start_time = time.time()
    #         self.infer_model.predict_on_batch(self.batch_input)
    #         result_time = time.time() - start_time

    #         summary = tf.Summary()
    #         summary_value = summary.value.add()
    #         summary_value.simple_value = result_time / self.fps_test_imgs_cnt
    #         summary_value.tag = "fps"
    #         self.tensorboard.writer.add_summary(summary, epoch)

class CustomLogger(Callback):
    def __init__(self,
                 config,
                 period=1,
                 tensorboard=None):

        self.period = period
        self.tensorboard = tensorboard
        self.config = config

        if not isinstance(self.tensorboard, TensorBoard) and self.tensorboard is not None:
            raise ValueError(
                "Tensorboard object must be a instance from keras.callbacks.TensorBoard")

    # def on_train_begin(self, logs=None):

    #     if self.tensorboard is not None and self.tensorboard.writer is not None:
    #         import json
    #         # with tf.Session() as sess:
    #         # hyperparameters = [tf.convert_to_tensor([k, str(v)]) for k, v in params.items()]
    #         tensor = tf.convert_to_tensor(json.dumps(self.config, indent=2))
    #         summary = tf.summary.text("Config", tensor)
    #         self.tensorboard.writer.add_summary(K.eval(summary))

    # def on_epoch_end(self, epoch, logs=None):

    #     if self.tensorboard is not None and self.tensorboard.writer is not None:
    #         lr = self.model.optimizer.lr
    #         decay = self.model.optimizer.decay
    #         iterations = self.model.optimizer.iterations
    #         lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))

    #         summary = tf.Summary()
    #         summary_value = summary.value.add()
    #         summary_value.simple_value = K.eval(lr_with_decay)
    #         summary_value.tag = "learning_rate"
    #         self.tensorboard.writer.add_summary(summary, epoch)


class MAP_evaluation(Callback):
    """ Evaluate a given dataset using a given model.
            code originally from https://github.com/fizyr/keras-retinanet

            # Arguments
                generator       : The generator that represents the dataset to evaluate.
                model           : The model to evaluate.
                iou_threshold   : The threshold used to consider when a detection is positive or negative.
                score_threshold : The score confidence threshold to use for detections.
                save_path       : The path to save images with visualized detections to.
            # Returns
                A dict mapping class names to mAP scores.
        """

    def __init__(self,
                 model,
                 generator,
                 period=1,
                 save_best=False,
                 save_name=None,
                 tensorboard=None,
                 ):
                 
        self.yolo_model = model
        self.generator = generator
        self.period = period
        self.save_best = save_best
        self.save_name_fmt = save_name
        self.tensorboard = tensorboard

        self.bestVloss = None
        self.bestMap = 0

        if not isinstance(self.tensorboard, TensorBoard) and self.tensorboard is not None:
            raise ValueError(
                "Tensorboard object must be a instance from keras.callbacks.TensorBoard")

    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.period == 0 and self.period != 0:
            average_precisions = self.yolo_model.evaluate_generator(generator=self.generator)
            print('\n')
            mAP = c_ut.print_predicted_average_precisions(average_precisions)

            # if not self.bestVloss:
            # self.bestVloss = logs['val_loss']

            if self.save_best and self.save_name_fmt:
                # and logs['val_loss'] <= self.bestVloss:
                if mAP > self.bestMap:
                    # self.bestVloss = logs['val_loss']
                    save_name = self.save_name_fmt.format(
                        epoch=epoch + 1, mAP=mAP, **logs)
                    print('\nEpoch %05d: mAP improved from %g to %g, saving model to %s' %
                          (epoch, self.bestMap, mAP, save_name))
                    self.bestMap = mAP
                    self.yolo_model.infer_model.save(save_name)
                else:
                    print("mAP did not improve from {}.".format(self.bestMap))

            if self.tensorboard is not None and self.tensorboard.writer is not None:
                import tensorflow as tf
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = mAP
                summary_value.tag = "val_mAP"
                self.tensorboard.writer.add_summary(summary, epoch)
