# coding: utf-8
import tensorflow as tf
import os
import random
import sys
from PIL import Image
import numpy as np

# 随机种子
_RANDOM_SEED = 0

# 数据集路径
train_imgs4_DIR = "train_imgs_4/"
train_imgs6_DIR = "train_imgs_6/"
test_imgs4_DIR = "test_imgs_4/"
test_imgs6_DIR = "test_imgs_6/"

# tfrecord文件存放路径
TFRECORD_DIR = "num_label_tf/"

# 判断tfrecord文件是否存在
def _dataset_exists(dataset_dir):
    for split_name in ['train', 'test']:
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


def image_to_tfexample(image_data, label):
    # Abstract base class for protocol messages.
    return tf.train.Example(features=tf.train.Features(feature={
        'image': bytes_feature(image_data),
        'label': int64_feature(label),
    }))


# 把数据转为TFRecord格式
def _convert_dataset(split_name, filenames):
    assert split_name in ['train', 'test']
    with tf.Session() as sess:
        # 定义tfrecord文件的路径+名字
        output_filename = os.path.join(TFRECORD_DIR, split_name + '.tfrecords')
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

                    # 获取label
                    labels = filename.split('/')[-1].split('.')[0]
                    label = len(labels)
                    if label == 4:
                        out = 0
                    else:
                        out = 1

                    # 生成protocol数据类型
                    example = image_to_tfexample(image_data, out)
                    tfrecord_writer.write(example.SerializeToString())

                except IOError as e:
                    print('Could not read:', filename)
                    print('Error:', e)
                    print('Skip it\n')
    sys.stdout.write('\n')
    sys.stdout.flush()

'''
# 判断tfrecord文件是否存在
if _dataset_exists(TFRECORD_DIR):
    print('tfcecord文件已存在')
else:
    # 获得所有图片路径,返回值为列表
    train_4num_filenames = _get_filenames_and_classes(train_imgs4_DIR)
    train_6num_filenames = _get_filenames_and_classes(train_imgs6_DIR)
    train_filenames = train_4num_filenames+train_6num_filenames
    print train_filenames
    random.seed(_RANDOM_SEED)
    # shuffle() 方法将序列的所有元素随机排序
    random.shuffle(train_filenames)
    print "训练集图片数量：%d"%len(train_filenames)

    test_4num_filenames = _get_filenames_and_classes(test_imgs4_DIR)
    test_6num_filenames = _get_filenames_and_classes(test_imgs6_DIR)
    test_filenames = test_4num_filenames + test_6num_filenames
    random.seed(_RANDOM_SEED)
    random.shuffle(test_filenames)
    print "测试集图片数量：%d" % len(test_filenames)

    # _convert_dataset('train', train_filenames)
    # _convert_dataset('test', test_filenames)
    # print('生成tfcecord文件')

'''

train_4num_filenames = _get_filenames_and_classes(train_imgs4_DIR)
train_6num_filenames = _get_filenames_and_classes(train_imgs6_DIR)
train_filenames = train_4num_filenames + train_6num_filenames
print train_filenames


