# coding: utf-8
import tensorflow as tf
import os
import sys
from PIL import Image
import numpy as np

charset = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
encode_maps = {}
decode_maps = {}
for i, char in enumerate(charset, 1):
    # i是从1-62的序号
    # char是0-Z字符
    encode_maps[char] = i-1 #根据字符对于序号
    decode_maps[i] = char #根据序号对于字符
print encode_maps
print decode_maps

# 数据集路径
train_imgs4_DIR = "train_imgs_4/"
train_imgs6_DIR = "train_imgs_6/"
test_imgs4_DIR = "test_imgs_4/"
test_imgs6_DIR = "test_imgs_6/"

# tfrecord文件存放路径
TFRECORD_DIR = "cmy_captcha/"

# 判断tfrecord文件是否存在
def _dataset_exists(dataset_dir):
    for split_name in ['train_4label', 'test_4label']:
        output_filename = os.path.join(dataset_dir, split_name + '.tfrecords')
        if not tf.gfile.Exists(output_filename):
            return False
    return True


# 获取所有验证码图片
def _get_filenames_and_classes(dataset_dir):
    photo_filenames = []
    for filename in os.listdir(dataset_dir):
        # 获取文件路径
        path = os.path.join(dataset_dir, filename)
        photo_filenames.append(path)
    return photo_filenames

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample4(image_data, label0, label1, label2, label3):
    # Abstract base class for protocol messages.
    return tf.train.Example(features=tf.train.Features(feature={
        'image': bytes_feature(image_data),
        'label0': int64_feature(label0),
        'label1': int64_feature(label1),
        'label2': int64_feature(label2),
        'label3': int64_feature(label3),
    }))

def image_to_tfexample6(image_data, label0, label1, label2, label3,label4,label5):
    # Abstract base class for protocol messages.
    return tf.train.Example(features=tf.train.Features(feature={
        'image': bytes_feature(image_data),
        'label0': int64_feature(label0),
        'label1': int64_feature(label1),
        'label2': int64_feature(label2),
        'label3': int64_feature(label3),
        'label4': int64_feature(label4),
        'label5': int64_feature(label5),
    }))

def gene_4num_tf(tfrecord_name, filenames):
    with tf.Session() as sess:
        # 定义tfrecord文件的路径+名字
        output_filename = os.path.join(TFRECORD_DIR, tfrecord_name + '.tfrecords')
        if not os.path.exists(TFRECORD_DIR):
            os.mkdir(TFRECORD_DIR)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i, filename in enumerate(filenames):
                # i为序数，filename为图片路径，如：5546 imges/6291.jpg
                try:
                    sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(filenames)))
                    sys.stdout.flush()
                    # 读取图片
                    image_data = Image.open(filename)
                    # 根据模型的结构resize
                    image_data = image_data.resize((224, 224))
                    # 灰度化
                    image_data = np.array(image_data.convert('L'))
                    # 将图片转化为bytes
                    image_data = image_data.tobytes()
                    # 将label列表化，如：['s', '1', 'n', 'X']，['O', 'L', 'S', 'A', 'N', 'Z']
                    labels = filename.split('/')[-1].split('.')[0]
                    labels_list = []
                    for c in labels:
                        label = encode_maps[c]
                        labels_list.append(label)
                    print labels_list
                    # 生成protocol数据类型
                    example = image_to_tfexample4(image_data, labels_list[0], labels_list[1], labels_list[2], labels_list[3])
                    tfrecord_writer.write(example.SerializeToString())
                except IOError as e:
                    print('Could not read:', filename)
                    print('Error:', e)
                    print('Skip it\n')
    sys.stdout.write('\n')
    sys.stdout.flush()

def gene_6num_tf(tfrecord_name, filenames):
    with tf.Session() as sess:
        # 定义tfrecord文件的路径+名字
        output_filename = os.path.join(TFRECORD_DIR, tfrecord_name + '.tfrecords')
        if not os.path.exists(TFRECORD_DIR):
            os.mkdir(TFRECORD_DIR)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i, filename in enumerate(filenames):
                # i为序数，filename为图片路径，如：5546 imges/6291.jpg
                try:
                    sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(filenames)))
                    sys.stdout.flush()
                    # 读取图片
                    image_data = Image.open(filename)
                    # 根据模型的结构resize
                    image_data = image_data.resize((224, 224))
                    # 灰度化
                    image_data = np.array(image_data.convert('L'))
                    # 将图片转化为bytes
                    image_data = image_data.tobytes()
                    # 将label列表化，如：['s', '1', 'n', 'X']，['O', 'L', 'S', 'A', 'N', 'Z']
                    labels = filename.split('/')[-1].split('.')[0]
                    labels_list = []
                    for c in labels:
                        label = encode_maps[c]
                        labels_list.append(label)
                    print labels_list
                    # 生成protocol数据类型
                    example = image_to_tfexample6(image_data, labels_list[0], labels_list[1], labels_list[2], labels_list[3],
                                                 labels_list[4],labels_list[5])
                    tfrecord_writer.write(example.SerializeToString())
                except IOError as e:
                    print('Could not read:', filename)
                    print('Error:', e)
                    print('Skip it\n')
    sys.stdout.write('\n')
    sys.stdout.flush()

# 判断tfrecord文件是否存在
if _dataset_exists(TFRECORD_DIR):
    print('tfcecord文件已存在')
else:
    # 获得训练集图片路径,返回值为列表,eg:train_imgs/00000824_yX15mL.png
    train_imgs4_name = _get_filenames_and_classes(train_imgs4_DIR)
    train_imgs6_name = _get_filenames_and_classes(train_imgs6_DIR)
    print "训练集图片4数量：%d"%len(train_imgs4_name),"训练集图片6数量：%d"%len(train_imgs6_name)
    test_imgs4_name = _get_filenames_and_classes(test_imgs4_DIR)
    test_imgs6_name = _get_filenames_and_classes(test_imgs6_DIR)
    print "测试集图片4数量：%d" % len(test_imgs4_name),"测试集图片6数量：%d" % len(test_imgs6_name)

    # 生成tfrecord文件
    gene_4num_tf('train_4label', train_imgs4_name)
    gene_6num_tf('train_6label', train_imgs6_name)
    gene_4num_tf('test_4label', test_imgs4_name)
    gene_6num_tf('test_6label', test_imgs6_name)
    print('生成tfcecord文件')

