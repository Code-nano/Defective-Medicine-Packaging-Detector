"""
Usage:
    python xml_to_tfrecord.py -traini /path/to/train/images -testi /path/to/test/images -trainxml /path/to/train/xmls -testxml /path/to/test/xmls -output /path/to/output/folder -label_map /path/to/label_map.pbtxt

Arguments:
    -traini, --train_image_dir : Path to the directory containing train images.
    -testi, --test_image_dir   : Path to the directory containing test images.
    -trainxml, --train_xml_dir : Path to the directory containing train XML files.
    -testxml, --test_xml_dir   : Path to the directory containing test XML files.
    -output, --output_dir      : Path to the output directory for CSV and TFRecord files (optional, default: current directory).
    -label_map, --label_map_file: Path to the label_map.pbtxt file.

Example:
    python xml_to_tfrecord.py -traini /data/train/images -testi /data/test/images -trainxml /data/train/xmls -testxml /data/test/xmls -output /data/output -label_map /data/label_map.pbtxt
"""
"""
Usage in Jupyter Notebook:
    !pyhton xml_to_tfrecord.py -traini /path/to/train/images -testi /path/to/test/images -trainxml /path/to/train/xmls -testxml /path/to/test/xmls -output /path/to/output/folder -label_map /path/to/label_map.pbtxt

Example:
    !python xml_to_tfrecord.py -traini /data/train/images -testi /data/test/images -trainxml /data/train/xmls -testxml /data/test/xmls -output /data/output -label_map /data/label_map.pbtxt
"""
"""
TF_RECORD_SCRIPT = files['TF_RECORD_SCRIPT']
TRAIN_IMAGE_DIR = os.path.join(paths['IMAGE_PATH'], 'train')
TEST_IMAGE_DIR = os.path.join(paths['IMAGE_PATH'], 'test')
TRAIN_XML_DIR = os.path.join(paths['ANNOTATION_PATH'], 'train')
TEST_XML_DIR = os.path.join(paths['ANNOTATION_PATH'], 'test')
OUTPUT_DIR = paths['ANNOTATION_PATH']

!python {TF_RECORD_SCRIPT} -traini {TRAIN_IMAGE_DIR} -testi {TEST_IMAGE_DIR} -trainxml {TRAIN_XML_DIR} -testxml {TEST_XML_DIR} -output {OUTPUT_DIR} -label_map {LABELMAP_DIR}
"""


import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
from object_detection.utils import dataset_util, label_map_util
import io
from PIL import Image
import argparse
from collections import namedtuple


def xml_to_csv(path):
    xml_list = []
    for root_dir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.xml'):
                xml_file = os.path.join(root_dir, file)
                tree = ET.parse(xml_file)
                xml_root = tree.getroot()

                for member in xml_root.findall('object'):
                    relative_path = os.path.relpath(os.path.dirname(xml_file), path) if os.path.dirname(xml_file) != '' else '.'
                    if not relative_path:  
                        relative_path = '.'  
                    value = (xml_root.find('filename').text,
                             int(xml_root.find('size')[0].text),
                             int(xml_root.find('size')[1].text),
                             member[0].text,
                             int(member[4][0].text),
                             int(member[4][1].text),
                             int(member[4][2].text),
                             int(member[4][3].text),
                             relative_path)
                    xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'relative_path']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    print(f"XML data:\n{xml_df}")
    return xml_df


def class_text_to_int(row_label, label_map):
    return label_map.get(row_label, None)

def parse_label_map(label_map_file):
    label_map = {}
    with open(label_map_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "name" in line:
                name = line.strip().split(":")[-1].strip().strip("'")
                id_line = lines[i + 1]
                id = int(id_line.strip().split(":")[-1].strip())
                label_map[name] = id
    return label_map

def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, group.relative_path, '{}'.format(group.filename)), 'rb') as fid:
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
        classes.append(class_text_to_int(row['class'], label_map))

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

def split(df, group_key):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group_key)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def generate_tfrecords(image_dir, csv_input, output_path):
    writer = tf.io.TFRecordWriter(output_path)
    path = os.path.join(image_dir)
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')

    processed = 0  # Add this line
    missing_images = 0

    for group_keys, group_data in grouped:
        for index, row in group_data.iterrows():
            try:
                tf_example = create_tf_example(row, image_dir)
                writer.write(tf_example.SerializeToString())
                processed += 1  # Add this line
                print(f"Processed {processed} records")  # Add this line
            except FileNotFoundError:
                missing_images += 1
                print(f"Missing image: {row['filename']}")

    writer.close()
    print(f"Successfully created the TFRecords: {output_path}")
    print(f"Processed {processed} records")  # Add this line
    print(f"Missing images: {missing_images}")  # Add this line


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert XML files to TensorFlow Record files')
    parser.add_argument('-traini', '--train_image_dir', dest='train_image_dir', type=str, required=True, help='Path to train images')
    parser.add_argument('-testi', '--test_image_dir', dest='test_image_dir', type=str, required=True, help='Path to test images')
    parser.add_argument('-trainxml', '--train_xml_dir', dest='train_xml_dir', type=str, required=True, help='Path to train XML files')
    parser.add_argument('-testxml', '--test_xml_dir', dest='test_xml_dir', type=str, required=True, help='Path to test XML files')
    parser.add_argument('-output', '--output_dir', dest='output_dir', type=str, default='.', help='Path to the output directory for CSV and TFRecord files (optional, default: current directory)')
    parser.add_argument('-label_map', '--label_map_file', dest='label_map_file', type=str, required=True, help='Path to the label_map.pbtxt file')

    args = parser.parse_args()
    
    # Parse the label_map.pbtxt file
    label_map = parse_label_map(args.label_map_file)
    
    # Convert XMLs to CSVs
    train_csv = xml_to_csv(args.train_xml_dir)
    test_csv = xml_to_csv(args.test_xml_dir)
    
    def find_relative_path(xml_dir, image_dir, filename):
        rel_path = os.path.dirname(os.path.relpath(os.path.join(image_dir, filename), image_dir))
        if not rel_path:
            rel_path = "."
            print(f"Filename: {filename}, XML Dir: {xml_dir}, Image Dir: {image_dir}, Relative Path: {rel_path}")
        return rel_path


    train_csv['relative_path'] = train_csv['filename'].apply(lambda x: find_relative_path(args.train_xml_dir, args.train_image_dir, x))
    test_csv['relative_path'] = test_csv['filename'].apply(lambda x: find_relative_path(args.test_xml_dir, args.test_image_dir, x))

    
    train_csv.to_csv(os.path.join(args.output_dir, 'train_labels.csv'), index=None)
    test_csv.to_csv(os.path.join(args.output_dir, 'test_labels.csv'), index=None)

    # Generate TFRecords
    generate_tfrecords(args.train_image_dir, os.path.join(args.output_dir, 'train_labels.csv'), os.path.join(args.output_dir, 'train.record'))
    generate_tfrecords(args.test_image_dir, os.path.join(args.output_dir, 'test_labels.csv'), os.path.join(args.output_dir, 'test.record'))
    print("Successfully created train.record and test.record in", args.output_dir)


