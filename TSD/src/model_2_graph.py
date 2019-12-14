import tensorflow as tf
import tensorflow.keras.backend as K
from _common.utils import makedirs
import os
import json
import yolo
from tensorflow.keras.models import Model, load_model
import shutil
from _common import utils
from pathlib import Path


def parse_args():
    import argparse
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-w', '--weights', help='weights path')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--ir', action='store_true', help='enable IR generation')
    argparser.add_argument('-t', '--trt', action='store_true', help='enable TRT engine generation')
    return argparser.parse_args()


def _main_():
    args = parse_args()
    
    weights_path = args.weights
    config_path = args.conf
    ir_flag = args.ir
    trt_flag = args.trt

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    output_pb_fpath = utils.get_pb_graph_fpath(config)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto())

    K.set_session(sess)
    K.set_learning_phase(0)

    train_sz = config['model']['infer_shape']

    if config['model'].get('labels'):
        labels = config['model'].get('labels')
    else:
        labels = ['object']

    config['model']['labels'] = labels
    yolo_model = yolo.YOLO_Model(
        config['model']
    )

    if weights_path:
        yolo_model.load_weights(weights_path)

    infer_model = yolo_model.infer_model

        # infer_model = load_model(weights_path)
        # image_input = Input(shape=(train_sz[0], train_sz[1], 3), name='input_img')
        # infer_model = Model(image_input, infer_model(image_input))
    # else:

    if type(infer_model.input) is list:
        model_input_names = [inp.name.split(':')[0] for inp in infer_model.input]
    else:
        model_input_names = [infer_model.input.name.split(':')[0]]
        
    if type(infer_model.output) is list:
        model_output_names = [out.name.split(':')[0] for out in infer_model.output]
    else:
        model_output_names = [infer_model.output.name.split(':')[0]]
    
    print('Model:')
    print('  Inputs: {}'.format(model_input_names))
    print('  Outputs: {}'.format(model_output_names))

    config['model']['output_names'] = model_output_names
    config['model']['input_names'] = model_input_names

    with K.get_session() as sess:

        graphdef = sess.graph.as_graph_def()

        dirpath = os.path.join('logs/tf_export', config['model']['base'])

        shutil.rmtree(dirpath, ignore_errors=True)
        makedirs(dirpath)

        writer = tf.summary.FileWriter(dirpath, sess.graph)
        writer.close()

        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, model_output_names)
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

    frozen_graph_filename = output_pb_fpath
    with open(frozen_graph_filename, 'wb') as f:
        f.write(frozen_graph.SerializeToString())
    f.close()

    K.clear_session()
    print('Frozen graph done!')

    if trt_flag:
        try:
            import uff
            uff_imported = True
        except:
            uff_imported = False
            print('TensorRT environment not found')
            
        if uff_imported:
            output_folder = "_gen/uff_models"
            makedirs(output_folder)
            result_uff_fname = Path(frozen_graph_filename).name
            result_uff_fpath = os.path.join(output_folder, str(Path(result_uff_fname).with_suffix('.uff')))
            result_cfg_path = str(Path(output_folder) / Path(result_uff_fname).with_suffix('.json'))
            
            uff_model = uff.from_tensorflow(frozen_graph, model_output_names, output_filename=result_uff_fpath)

            with open(result_cfg_path, 'w') as f:
                json.dump(config, f, indent=4)
            
    if ir_flag:
        try:
            import mo_tf
            
            from subprocess import call
            openvino_found = True
        except ModuleNotFoundError:
            print('OpenVINO environment not found')
            openvino_found = False

        if openvino_found:
            output_folder = "_gen/ir_models"
            makedirs(output_folder)
            result_pb_fname = Path(frozen_graph_filename).name
            result_cfg_path = str(Path(output_folder) / Path(result_pb_fname).with_suffix('.json'))
            
            process_args = ["mo_tf.py", 
                            "--input_model", frozen_graph_filename, 
                            "--scale", "255",
                            "--model_name", str(Path(result_pb_fname).with_suffix(''))+"_FP16",
                            "--data_type", "FP16", 
                            "--input_shape", "[1,{},{},3]".format(*train_sz),
                            "--output_dir", output_folder]
            call(process_args)

            process_args = ["mo_tf.py", 
                            "--input_model", frozen_graph_filename, 
                            "--scale", "255",
                            "--model_name", str(Path(result_pb_fname).with_suffix(''))+"_FP32",
                            "--data_type", "FP32", 
                            "--input_shape", "[1,{},{},3]".format(*train_sz),
                            "--output_dir", output_folder]
            call(process_args)

            with open(result_cfg_path, 'w') as f:
                json.dump(config, f, indent=4)

        
if __name__ == '__main__':
    _main_()
