# coding: utf-8
import tensorflow as tf
import logging
logger = logging.getLogger()
# 不同字符数量
CHAR_SET_LEN = 2
# 图片高度
IMAGE_HEIGHT = 60
# 图片宽度
IMAGE_WIDTH = 160
# 批次
train_BATCH_SIZE = 32
test_BATCH_SIZE = 10000

lr = tf.Variable(0.01, dtype=tf.float32)

# tfrecord文件存放路径
TFRECORD_FILE = "num_label_tf/train.tfrecords"
TEST_FILE = "num_label_tf/test.tfrecords"

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
    label = tf.cast(features['label'], tf.int32)
    return image, label

# 获取训练集数据
image, label = read_and_decode(TFRECORD_FILE)
# 使用shuffle_batch可以随机打乱
image_batch, label_batch = tf.train.shuffle_batch(
    [image, label], batch_size=train_BATCH_SIZE,
    capacity=4000,min_after_dequeue = 2000, num_threads=1)

# 获取测试集数据
test_image, test_label = read_and_decode(TEST_FILE)
test_image_batch,test_label_batch = tf.train.shuffle_batch(
    [test_image, test_label], batch_size=test_BATCH_SIZE,
    capacity=3000,min_after_dequeue = 1000, num_threads=1)

sess = tf.Session()
# 初始化权值
def weight_variable(shape):
    # 生成一个截断的正态分布
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
# 初始化偏置值
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
# 卷基层
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
# 池化层
def pool(x):
    return tf.nn.max_pool(x,ksize=[1,4,4,1],strides=[1,4,4,1],padding="SAME")

# 输入
# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224])
y = tf.placeholder(tf.float32, [None])

x_img = tf.reshape(x, [-1, 224, 224, 1])
one_hot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=CHAR_SET_LEN)

# 第一个卷积层
W1 = weight_variable([5,5,1,32])
b1 = bias_variable([32])
L1 = tf.nn.relu(conv2d(x_img,W1)+b1)
pool1 = pool(L1)# 56*56**32

# 第二个卷积层
W2 = weight_variable([5,5,32,64])
b2 = bias_variable([64])
L2 = tf.nn.relu(conv2d(pool1,W2)+b2)
pool2 = pool(L2) # 14*14*64

# 全连接层
x_input = tf.reshape(pool2,[-1,14*14*64])
W3 = weight_variable([14*14*64,1024])
b3 = bias_variable([1024])
keep_prob = tf.placeholder(tf.float32)
L3 = tf.nn.relu(tf.matmul(x_input,W3)+b3)
L3_prob = tf.nn.dropout(L3,keep_prob=keep_prob)

W4 = weight_variable([1024,2])
b4 = bias_variable([2])
prediction = tf.nn.softmax(tf.matmul(L3,W4)+b4)

# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=one_hot_labels))
# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction),axis=1))
train = tf.train.AdadeltaOptimizer(learning_rate=lr).minimize(loss)

#accaurcy
one_hot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=CHAR_SET_LEN)
correct_pre = tf.equal(tf.argmax(one_hot_labels,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

'''
#保存模型
saver = tf.train.Saver()
# 训练过程，使用了交互式session
sess.run(tf.global_variables_initializer())
# 创建一个协调器，管理线程
coord = tf.train.Coordinator()
# 启动QueueRunner, 此时文件名队列已经进队
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for i in range(80001):
    b_image, b_label= sess.run([image_batch, label_batch])
    t_image, t_label = sess.run([test_image_batch, test_label_batch])
    sess.run(train,feed_dict={x: b_image, y: b_label, keep_prob: 0.6})
    if i%20 ==0:
        if i % 5000 == 0:
            sess.run(tf.assign(lr, lr / 2))
        loss_ = sess.run(loss,feed_dict={x: b_image, y: b_label, keep_prob: 0.6})
        train_acc = sess.run(accuracy,feed_dict={x: b_image, y: b_label, keep_prob: 0.6})
        test_acc = sess.run(accuracy, feed_dict={x: t_image, y: t_label, keep_prob: 0.6})
        print 'Step %d ,loss %2f, training accuracy %3f ,testing accuracy %3f' % (i,loss_, train_acc, test_acc)
        learning_rate = sess.run(lr)
    if i == 80000:
        saver.save(sess, "./num_label_tf/models/label_num.model", global_step=i)
        break
# 通知其他线程关闭
coord.request_stop()
# 其他所有线程关闭之后，这一函数才能返回
coord.join(threads)
'''

'''
80000次，batch_size = 32,lr = 0.01,AdadeltaOptimizer,交叉熵，结果：loss：0.34,train acc:很高,test acc:0.986
'''

#加载测试好的模型进行测试
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
# 启动QueueRunner, 此时文件名队列已经进队
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
t_image, t_label = sess.run([test_image_batch, test_label_batch])
saver.restore(sess,'num_label_tf/models/label_num.model-80000')
print(sess.run(accuracy,feed_dict={x:t_image,y:t_label}))

'''
测试：1万张acc：0.9773
'''




