from keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
import numpy as np
import keras
from utils.utils import evaluate


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
                            self.model_to_save.save_weights(filepath, overwrite=True)
                        else:
                            self.model_to_save.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model_to_save.save_weights(filepath, overwrite=True)
                else:
                    self.model_to_save.save(filepath, overwrite=True)

        super(CustomModelCheckpoint, self).on_batch_end(epoch, logs)


class MAP_evaluation(keras.callbacks.Callback):
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
                 infer_model,
                 generator,
                 iou_threshold=0.3,
                 score_threshold=0.3,
                 save_path=None,
                 period=1,
                 save_best=False,
                 save_name=None,
                 tensorboard=None,
                 infer_sz=[416, 416]):

        self.infer_model = infer_model
        self.generator = generator
        self.iou_threshold = iou_threshold
        self.save_path = save_path
        self.period = period
        self.save_best = save_best
        self.save_name_fmt = save_name
        self.tensorboard = tensorboard
        self.infer_sz = infer_sz

        self.bestVloss = None
        self.bestMap = 0

        if not isinstance(self.tensorboard, keras.callbacks.TensorBoard) and self.tensorboard is not None:
            raise ValueError("Tensorboard object must be a instance from keras.callbacks.TensorBoard")

    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.period == 0 and self.period != 0:
            average_precisions = evaluate(model=self.infer_model,
                                          generator=self.generator,
                                          iou_threshold=self.iou_threshold,
                                          net_h=self.infer_sz[0],
                                          net_w=self.infer_sz[1],
                                          save_path=None)
            print('\n')
            for label, average_precision in average_precisions.items():
                print(label, '{:.4f}'.format(average_precision))
            mAP = sum(average_precisions.values()) / len(average_precisions)
            print('mAP: {:.4f}'.format(mAP))

            if not self.bestVloss:
                self.bestVloss = logs['val_loss']

            if self.save_best and self.save_name_fmt:
                if mAP >= self.bestMap and logs['val_loss'] =< self.bestVloss:
                    self.bestVloss = logs['val_loss']
                    self.bestMap = mAP

                    save_name = self.save_name_fmt.format(epoch=epoch + 1, mAP=mAP, **logs)
                    print('\nEpoch %05d: mAP improved from %g to %g, saving model to %s' % (epoch, self.bestMap, mAP,
                                                                                            save_name))
                    self.infer_model.save(save_name)
                else:
                    print("mAP did not improve from {}.".format(self.bestMap))

            if self.tensorboard is not None and self.tensorboard.writer is not None:
                import tensorflow as tf
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = mAP
                summary_value.tag = "val_mAP"
                self.tensorboard.writer.add_summary(summary, epoch)


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
