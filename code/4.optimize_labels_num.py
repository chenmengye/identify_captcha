# coding: utf-8
import tensorflow as tf
from nets_label import nets_factory

# 不同字符数量
CHAR_SET_LEN = 2
# 图片高度
IMAGE_HEIGHT = 60
# 图片宽度
IMAGE_WIDTH = 160
# 批次
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 100
# tfrecord文件存放路径
TRAIN_FILE = "num_label_tf/train.tfrecords"
TEST_FILE= "num_label_tf/test.tfrecords"

# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224])
y = tf.placeholder(tf.float32, [None])

# 学习率
lr = tf.Variable(0.02, dtype=tf.float32)

# 从tfrecord读出数据
def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })
    # 获取图片数据
    image = tf.decode_raw(features['image'], tf.uint8)
    # tf.train.shuffle_batch必须确定shape
    image = tf.reshape(image, [224, 224])
    # 图片预处理
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    # 获取label
    label0 = tf.cast(features['label'], tf.int32)
    return image, label0

# 获取训练图片数据和标签
train_image,train_label = read_and_decode(TRAIN_FILE)
# 使用shuffle_batch可以随机打乱
image_batch, label_batch = tf.train.shuffle_batch(
    [train_image, train_label], batch_size=TRAIN_BATCH_SIZE,
    capacity=40000, min_after_dequeue=20000, num_threads=1)

# 获取测试图片数据和标签
test_image,test_label = read_and_decode(TEST_FILE)
# 使用shuffle_batch可以随机打乱
t_image_batch, t_label_batch = tf.train.shuffle_batch(
    [test_image, test_label], batch_size=TEST_BATCH_SIZE,
    capacity=20000, min_after_dequeue=8000, num_threads=1)

# 定义网络结构
train_network_fn = nets_factory.get_network_fn(
    'alexnet_v2',
    num_classes=CHAR_SET_LEN,
    weight_decay=0.0005,
    is_training=True)

with tf.Session() as sess:
    X = tf.reshape(x, [-1, 224, 224, 1])
    # 数据输入网络得到输出值
    logits,end_points = train_network_fn(X)
    # 把标签转成one_hot的形式
    one_hot_labels0 = tf.one_hot(indices=tf.cast(y, tf.int32), depth=CHAR_SET_LEN)
    # 计算loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels0))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    # 计算准确率
    correct_prediction = tf.equal(tf.argmax(one_hot_labels0, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 用于保存模型
    saver = tf.train.Saver()
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    # 启动QueueRunner, 此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(50001):
        # 获取训练数据一个批次的数据和标签
        b_image, b_label = sess.run(
            [image_batch, label_batch])
        # 优化模型
        sess.run(optimizer, feed_dict={x: b_image, y: b_label})
        # 每迭代20次计算一次loss和准确率
        if i % 20 == 0:
            # 每迭代2000次降低一次学习率
            if i % 2000 == 0:
                sess.run(tf.assign(lr, lr / 2))
                # sess.run(tf.assign(lr, 0.001 * (0.95 ** i)))
            acc, loss_ = sess.run([accuracy,loss],feed_dict={x: b_image,y: b_label})
            learning_rate = sess.run(lr)
            print ("Iter:%d  Loss:%.3f  train accuracy:%.2f Learning_rate:%.4f" % (
            i, loss_, acc,  learning_rate))
            # 获取训练数据一个批次的数据和标签
            t_image, t_label = sess.run([t_image_batch, t_label_batch,])
            t_acc = sess.run(accuracy, feed_dict={x: t_image,y: t_label})
            print ("Iter:%d test accuracy:%.2f " % (i, t_acc))
            if i == 50000:
                saver.save(sess, "./num_label_tf/optimize_models/label_num.model", global_step=i)
                break
    # 通知其他线程关闭
    coord.request_stop()
    # 其他所有线程关闭之后，这一函数才能返回
    coord.join(threads)

