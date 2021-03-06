# coding: utf-8
import tensorflow as tf
from nets_6f import nets_factory

# 不同字符数量
CHAR_SET_LEN = 62
# 图片高度
IMAGE_HEIGHT = 60
# 图片宽度
IMAGE_WIDTH = 160
# 批次
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 100
# tfrecord文件存放路径
TRAIN_FILE = "cmy_captcha/train_6label.tfrecords"
TEST_FILE= "cmy_captcha/test_6label.tfrecords"

# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224])
y0 = tf.placeholder(tf.float32, [None])
y1 = tf.placeholder(tf.float32, [None])
y2 = tf.placeholder(tf.float32, [None])
y3 = tf.placeholder(tf.float32, [None])
y4 = tf.placeholder(tf.float32, [None])
y5 = tf.placeholder(tf.float32, [None])

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
                                           'label0': tf.FixedLenFeature([], tf.int64),
                                           'label1': tf.FixedLenFeature([], tf.int64),
                                           'label2': tf.FixedLenFeature([], tf.int64),
                                           'label3': tf.FixedLenFeature([], tf.int64),
                                           'label4': tf.FixedLenFeature([], tf.int64),
                                           'label5': tf.FixedLenFeature([], tf.int64),
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
    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)
    label4 = tf.cast(features['label4'], tf.int32)
    label5 = tf.cast(features['label5'], tf.int32)
    return image, label0, label1, label2, label3, label4, label5

# 获取训练图片数据和标签
train_image,train_label0,train_label1,train_label2,train_label3,train_label4,train_label5 = read_and_decode(TRAIN_FILE)
# 使用shuffle_batch可以随机打乱
image_batch, label_batch0, label_batch1, label_batch2, label_batch3, label_batch4, label_batch5 = tf.train.shuffle_batch(
    [train_image, train_label0, train_label1, train_label2, train_label3,train_label4,train_label5], batch_size=TRAIN_BATCH_SIZE,
    capacity=40000, min_after_dequeue=20000, num_threads=1)

# 获取测试图片数据和标签
test_image,test_label0,test_label1,test_label2,test_label3,test_label4,test_label5 = read_and_decode(TEST_FILE)
# 使用shuffle_batch可以随机打乱
t_image_batch, t_label_batch0, t_label_batch1, t_label_batch2, t_label_batch3, t_label_batch4, t_label_batch5 = tf.train.shuffle_batch(
    [test_image, test_label0, test_label1, test_label2, test_label3, test_label4, test_label5], batch_size=TEST_BATCH_SIZE,
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
    logits0, logits1, logits2, logits3, logits4, logits5, end_points = train_network_fn(X)
    # 把标签转成one_hot的形式
    one_hot_labels0 = tf.one_hot(indices=tf.cast(y0, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labels1 = tf.one_hot(indices=tf.cast(y1, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labels2 = tf.one_hot(indices=tf.cast(y2, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labels3 = tf.one_hot(indices=tf.cast(y3, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labels4 = tf.one_hot(indices=tf.cast(y4, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labels5 = tf.one_hot(indices=tf.cast(y5, tf.int32), depth=CHAR_SET_LEN)
    # 计算loss
    loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits0, labels=one_hot_labels0))
    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=one_hot_labels1))
    loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=one_hot_labels2))
    loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits3, labels=one_hot_labels3))
    loss4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits4, labels=one_hot_labels4))
    loss5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits5, labels=one_hot_labels5))

    # 计算总的loss
    total_loss = (loss0 + loss1 + loss2 + loss3 + loss4 + loss5) / 6.0
    # 优化total_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)
    # 计算准确率
    correct_prediction0 = tf.equal(tf.argmax(one_hot_labels0, 1), tf.argmax(logits0, 1))
    accuracy0 = tf.reduce_mean(tf.cast(correct_prediction0, tf.float32))

    correct_prediction1 = tf.equal(tf.argmax(one_hot_labels1, 1), tf.argmax(logits1, 1))
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

    correct_prediction2 = tf.equal(tf.argmax(one_hot_labels2, 1), tf.argmax(logits2, 1))
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))

    correct_prediction3 = tf.equal(tf.argmax(one_hot_labels3, 1), tf.argmax(logits3, 1))
    accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))

    correct_prediction4 = tf.equal(tf.argmax(one_hot_labels4, 1), tf.argmax(logits4, 1))
    accuracy4 = tf.reduce_mean(tf.cast(correct_prediction4, tf.float32))

    correct_prediction5 = tf.equal(tf.argmax(one_hot_labels5, 1), tf.argmax(logits5, 1))
    accuracy5 = tf.reduce_mean(tf.cast(correct_prediction5, tf.float32))

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
        b_image, b_label0, b_label1, b_label2, b_label3, b_label4, b_label5 = sess.run(
            [image_batch, label_batch0, label_batch1, label_batch2, label_batch3, label_batch4, label_batch5])
        # 优化模型
        sess.run(optimizer, feed_dict={x: b_image, y0: b_label0, y1: b_label1, y2: b_label2, y3: b_label3,  y4: b_label4, y5: b_label5})
        # 每迭代20次计算一次loss和准确率
        if i % 20 == 0:
            # 每迭代2000次降低一次学习率
            if i % 4000 == 0:
                sess.run(tf.assign(lr, lr / 2))
                # sess.run(tf.assign(lr, 0.01 * (0.95 ** i)))
            t_acc0, t_acc1, t_acc2, t_acc3, t_acc4, t_acc5, loss_ = sess.run([accuracy0, accuracy1, accuracy2, accuracy3,
                                                                              accuracy4, accuracy5,total_loss],
                                                     feed_dict={x: b_image,
                                                                y0: b_label0,
                                                                y1: b_label1,
                                                                y2: b_label2,
                                                                y3: b_label3,
                                                                y4: b_label4,
                                                                y5: b_label5
                                                                })
            learning_rate = sess.run(lr)
            print ("Iter:%d  Loss:%.3f  train accuracy:%.2f,%.2f,%.2f,%.2f,%.2f,%.2f  Learning_rate:%.4f" % (
            i, loss_, t_acc0, t_acc1, t_acc2, t_acc3, t_acc4, t_acc5, learning_rate))
            # 获取训练数据一个批次的数据和标签
            t_image, t_label0, t_label1, t_label2, t_label3, t_label4, t_label5 = sess.run([t_image_batch, t_label_batch0,
                                                                       t_label_batch1, t_label_batch2, t_label_batch3,
                                                                       t_label_batch4,t_label_batch5])
            acc0, acc1, acc2, acc3, acc4, acc5 = sess.run([accuracy0, accuracy1, accuracy2, accuracy3, accuracy4, accuracy5,],
                                                     feed_dict={x: t_image,
                                                                y0: t_label0,
                                                                y1: t_label1,
                                                                y2: t_label2,
                                                                y3: t_label3,
                                                                y4: t_label4,
                                                                y5: t_label5
                                                                })
            print ("Iter:%d test accuracy:%.2f,%.2f,%.2f,%.2f,%.2f,%.2f " % (
                i, acc0, acc1, acc2, acc3, acc2, acc3))
            if i == 50000:
                saver.save(sess, "./cmy_captcha/models/6labels_captcha.model", global_step=i)
                break
    # 通知其他线程关闭
    coord.request_stop()
    # 其他所有线程关闭之后，这一函数才能返回
    coord.join(threads)

