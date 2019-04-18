from utils import *
from tqdm import tqdm

dst_shape_img = (640, 320)


def get_samples(rootdir):

    png_fpaths = []

    for root, subdirs, files in os.walk(rootdir):
        png_fpaths += [os.path.join(root, file) for file in files if file.endswith('.png')]

    annotation_pairs = [(file, file.replace('.png', '_color_mask.png')) for file in png_fpaths if os.path.exists(file.replace('.png', '_color_mask.png'))]

    orig_imgs = []
    lane_imgs = []

    upper_clip_px = 50

    for src_file, msk_file in tqdm(annotation_pairs):

        orig_img = cv2.imread(src_file)
        mask_img = cv2.imread(msk_file)

        img_h, img_w, _ = mask_img.shape
        
        orig_img = orig_img[1+upper_clip_px:img_h-1, 1:img_w-1]
        mask_img = mask_img[1+upper_clip_px:img_h-1, 1:img_w-1]

        orig_img = cv2.resize(orig_img, dst_shape_img);
        mask_img = cv2.resize(mask_img, dst_shape_img, interpolation=cv2.INTER_NEAREST);

        layer_colors = set( tuple(v) for m2d in mask_img for v in m2d )
        lane_color = (250, 250, 250)
        
        if lane_color not in layer_colors:
            print('Error')
            exit(1)

        lane_layer = cv2.inRange(mask_img, lane_color, lane_color)

        orig_imgs += [orig_img]
        lane_imgs += [lane_layer]

    orig_imgs = np.array(orig_imgs)
    # Extend to 4 dims
    lane_imgs = np.expand_dims(lane_imgs, axis=-1)

    if len(orig_imgs.shape) != 4 or len(lane_imgs.shape) != 4:
        raise Exception('Invalid data shape')

    return (orig_imgs, lane_imgs)


def preprocess_data(inputs, outputs, input_shp, output_shp):
    data_count = len(inputs)

    input_imgs_shp = tuple([data_count] + list(input_shp))
    output_imgs_shp = tuple([data_count] + list(output_shp))

    new_inputs  = np.ndarray(input_imgs_shp, dtype=np.float32)
    new_outputs = np.ndarray(output_imgs_shp, dtype=np.float32)

    for i in range(data_count):
        new_inputs[i]  = image_preprocess(inputs[i], (input_shp[1], input_shp[0]))
        new_output = mask_preprocess(outputs[i], (output_shp[1], output_shp[0]))
        new_outputs[i] = np.expand_dims(new_output, axis=-1)

    return (new_inputs, new_outputs)

###############################################################################

def robofest_data_get_samples():

    train_rootdir = '../data/robofest_18_lanes'
    valid_rootdir = '../data/robofest_18_test'
    
    print('Processing train data')
    train_imgs, train_masks = get_samples(train_rootdir)
    
    print('Processing test data')
    valid_imgs, valid_masks = get_samples(valid_rootdir)

    return train_imgs, train_masks, valid_imgs, valid_masks


def robofest_data_get_samples_preprocessed(input_shp, output_shp):

    train_imgs, train_masks, valid_imgs, valid_masks = robofest_data_get_samples()

    train_imgs, train_masks = preprocess_data(train_imgs, train_masks, input_shp, output_shp)
    valid_imgs, valid_masks = preprocess_data(valid_imgs, valid_masks, input_shp, output_shp)

    return train_imgs, train_masks, valid_imgs, valid_masks
