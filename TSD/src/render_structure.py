from _common.utils import init_session
import argparse
import yolo 
import json
from tensorflow.keras.utils import plot_model
import os

def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    init_session(0.3)

    labels = ['sign']
    config['model']['labels'] = labels
    yolo_model = yolo.YOLO_Model(config['model'])

    model_render_file = 'images/{}.png'.format(config['model']['base'])
    if not os.path.isdir(os.path.dirname(model_render_file)):
        os.makedirs(os.path.dirname(model_render_file))
    
    print(yolo_model.infer_model.summary())
        
    plot_model(yolo_model.infer_model, to_file=model_render_file, show_shapes=True)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Help =)')
    argparser.add_argument('conf', help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)

