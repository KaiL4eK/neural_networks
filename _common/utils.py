import os


def get_impaths_from_path(path):

    image_paths = []

    if os.path.isdir(path):
        for root, subdirs, files in os.walk(path):
            pic_extensions = ('.png', '.PNG', '.jpg', 'JPEG', '.ppm')
            image_paths += [os.path.join(root, file) for file in files if file.endswith(pic_extensions)]
    else:
        image_paths += [path]

    return image_paths

def data_generator(path):

    if '/dev/video' in input_path: # do detection on the first webcam
        video_reader = cv2.VideoCapture(input_path)

        # the main loop
        batch_size  = 1
        images      = []
        while True:
            ret_val, image = video_reader.read()

        	yield image;
        
    elif input_path[-4:] == '.mp4' or input_path[-5:] == '.webm': # do detection on a video  
        
        video_reader = cv2.VideoCapture(input_path)

        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            yield image;

        video_reader.release()

    else: # do detection on an image or a set of images
        image_paths = get_impaths_from_path(input_path)

        # the main loop
        for image_path in tqdm(image_paths):
            image = cv2.imread(image_path)
            
            yield image
