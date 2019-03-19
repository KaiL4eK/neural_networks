import cv2
import os
import tqdm

import argparse
argparser = argparse.ArgumentParser(description='Help =)')
argparser.add_argument('-i', '--input', help='path to folder to flip')
argparser.add_argument('-o', '--output', help='path to folder to output')


def main(args):
    inputFldr = args.input
    outputFldr = args.output

    if not os.path.exists(outputFldr):
        os.mkdir(outputFldr)
        print("Directory {} created".format(outputFldr))
    else:
        print("Directory {} already exists".format(outputFldr))

    image_fpaths = []
    image_fnames = []

    if os.path.isdir(inputFldr): 
        for inp_file in os.listdir(inputFldr):
            image_fpaths += [os.path.join(inputFldr, inp_file)]
            image_fnames += [inp_file]
    else:
        print('--input must be set to directory')

    for i, image_path in enumerate(image_fpaths):
        img = cv2.imread(image_path)

        img = cv2.flip(img, 1);

        outputFname = os.path.join(outputFldr, image_fnames[i])

        print(outputFname)

        cv2.imwrite(outputFname, img)


if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)
