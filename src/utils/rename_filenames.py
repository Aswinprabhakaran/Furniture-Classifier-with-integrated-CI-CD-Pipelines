import os
import argparse

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('-img_path', '--images_path', required = True , type = str , help = 'path to images to rename')
    ap.add_argument('-prefix', '--prefix', required = True , type = str , help = 'prefiX text to include')
    args = vars(ap.parse_args())
    print(args)

    image_list = os.listdir(args['images_path'])

    for index, fname in enumerate(image_list):

        index += 1

        new_name = "{}_{}.jpg".format(args['prefix'].lower(), index)

        os.rename(os.path.join(args['images_path'], fname), os.path.join(args['images_path'], new_name))

        if index % 10 == 0:
            print("Processed : {}/{}".format(index, len(image_list)))