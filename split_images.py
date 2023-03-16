import os
import shutil
import random
import argparse
import logging
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Split image dataset into training and testing sets.')
    parser.add_argument('--data_folder', type=str, default=os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages'),
                        help='Path to the folder containing the image dataset')
    parser.add_argument('--train_folder', type=str, default=os.path.join('Tensorflow', 'workspace', 'images', 'train'),
                        help='Path to the output train folder')
    parser.add_argument('--test_folder', type=str, default=os.path.join('Tensorflow', 'workspace', 'images', 'test'),
                        help='Path to the output test folder')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                        help='Ratio of images to be used for training (e.g., 0.8 means 80% for training and 20% for testing)')
    return parser.parse_args()

def get_images_by_subfolder(data_folder):
    images_by_subfolder = {}
    for folder in os.listdir(data_folder):
        images = [os.path.join(data_folder, folder, f) for f in os.listdir(os.path.join(data_folder, folder)) if f.endswith(".jpg")]
        images_by_subfolder[folder] = images
    return images_by_subfolder

def split_and_copy_images(train_folder, test_folder, images_by_subfolder, split_ratio):
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for folder, images in images_by_subfolder.items():
        random.shuffle(images)
        train_images = images[:int(split_ratio * len(images))]
        test_images = images[int(split_ratio * len(images)):]

        for image_path in tqdm(train_images, desc=f"Copying train images from {folder}"):
            image_name = os.path.basename(image_path)
            xml_name = os.path.splitext(image_name)[0] + ".xml"
            xml_path = os.path.join(os.path.dirname(image_path), xml_name)
            shutil.copy(image_path, os.path.join(train_folder, image_name))
            shutil.copy(xml_path, os.path.join(train_folder, xml_name))

        for image_path in tqdm(test_images, desc=f"Copying test images from {folder}"):
            image_name = os.path.basename(image_path)
            xml_name = os.path.splitext(image_name)[0] + ".xml"
            xml_path = os.path.join(os.path.dirname(image_path), xml_name)
            shutil.copy(image_path, os.path.join(test_folder, image_name))
            shutil.copy(xml_path, os.path.join(test_folder, xml_name))

def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()

    if not os.path.exists(args.data_folder):
        logging.error(f"Data folder '{args.data_folder}' does not exist.")
        return

    images_by_subfolder = get_images_by_subfolder(args.data_folder)

    if not images_by_subfolder:
        logging.error("No images found in the data folder.")
        return

    split_and_copy_images(args.train_folder, args.test_folder, images_by_subfolder, args.split_ratio)
    logging.info("Successfully split and copied images and XML files to train and test folders.")

if __name__ == "__main__":
    main()
