"""
Usage:
    python xml_to_tfrecord.py -traini /path/to/train/images -testi /path/to/test/images -trainxml /path/to/train/xmls -testxml /path/to/test/xmls -output /path/to/output/folder

Arguments:
    -traini, --train_image_dir : Path to the directory containing train images.
    -testi, --test_image_dir   : Path to the directory containing test images.
    -trainxml, --train_xml_dir : Path to the directory containing train XML files.
    -testxml, --test_xml_dir   : Path to the directory containing test XML files.
    -output, --output_dir      : Path to the output directory for CSV and TFRecord files (optional, default: current directory).

Example:
    python xml_to_tfrecord.py -traini /data/train/images -testi /data/test/images -trainxml /data/train/xmls -testxml /data/test/xmls -output /data/output
"""
"""
Usage in Jupyter Notebook:
    %run xml_to_tfrecord.py -traini /path/to/train/images -testi /path/to/test/images -trainxml /path/to/train/xmls -testxml /path/to/test/xmls -output /path/to/output/folder

Example:
    %run xml_to_tfrecord.py -traini /data/train/images -testi /data/test/images -trainxml /data/train/xmls -testxml /data/test/xmls -output /data/output
"""
"""
TF_RECORD_SCRIPT = files['TF_RECORD_SCRIPT']
TRAIN_IMAGE_DIR = os.path.join(paths['IMAGE_PATH'], 'train')
TEST_IMAGE_DIR = os.path.join(paths['IMAGE_PATH'], 'test')
TRAIN_XML_DIR = os.path.join(paths['ANNOTATION_PATH'], 'train')
TEST_XML_DIR = os.path.join(paths['ANNOTATION_PATH'], 'test')
OUTPUT_DIR = paths['ANNOTATION_PATH']

%run {TF_RECORD_SCRIPT} -traini {TRAIN_IMAGE_DIR} -testi {TEST_IMAGE_DIR} -trainxml {TRAIN_XML_DIR} -testxml {TEST_XML_DIR} -output {OUTPUT_DIR}
"""


import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
from object_detection.utils import dataset_util
import io
from PIL import Image
import argparse

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def class_text_to_int(row_label):
    if row_label == 'your_class_name':
        return 1
    else:
        return None

def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def generate_tfrecords(image_dir, csv_input, output_path):
    writer = tf.io.TFRecordWriter(output_path)
    path = os.path.join(image_dir)
    examples = pd.read_csv(csv_input)
    grouped = examples.groupby('filename')

    for group in grouped.groups.keys():
        group_data = grouped.get_group(group)
        tf_example = create_tf_example(group_data, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print(f'Successfully created the {output_path} file')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert XML files to TensorFlow Record files')
    parser.add_argument('-traini', '--train_image_dir', dest='train_image_dir', type=str, required=True, help='Path to train images')
    parser.add_argument('-testi', '--test_image_dir', dest='test_image_dir', type=str, required=True, help='Path to test images')
    parser.add_argument('-trainxml', '--train_xml_dir', dest='train_xml_dir', type=str, required=True, help='Path to train XML files')
    parser.add_argument('-testxml', '--test_xml_dir', dest='test_xml_dir', type=str, required=True, help='Path to test XML files')
    parser.add_argument('-output', '--output_dir', dest='output_dir', type=str, default='.', help='Path to the output directory for CSV and TFRecord files (optional, default: current directory)')

    args = parser.parse_args()

    # Convert XMLs to CSVs
    train_csv = xml_to_csv(args.train_xml_dir)
    test_csv = xml_to_csv(args.test_xml_dir)
    train_csv.to_csv(os.path.join(args.output_dir, 'train_labels.csv'), index=None)
    test_csv.to_csv(os.path.join(args.output_dir, 'test_labels.csv'), index=None)
    print("Successfully created train_labels.csv and test_labels.csv in", args.output_dir)

    # Generate TFRecords
    generate_tfrecords(args.train_image_dir, os.path.join(args.output_dir, 'train_labels.csv'), os.path.join(args.output_dir, 'train.record'))
    generate_tfrecords(args.test_image_dir, os.path.join(args.output_dir, 'test_labels.csv'), os.path.join(args.output_dir, 'test.record'))
    print("Successfully created train.record and test.record in", args.output_dir)


