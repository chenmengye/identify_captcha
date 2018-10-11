# different_words_captcha
identify four_chars captcha or six_chars captcha
</br>
### <font color=#00FFFF >首先简单介绍一下整体的思路,由于要识别的验证码有4位和6位，所以先训练一个模型来识别验证码的字符个数，再经过个数识别来确定用4位数模型还是6位数模型</font>
#### 1.gen_img.py 生成验证码图片</br>
四位数验证码训练图片：80000张，测试图片：20000张；六位数验证码训练图片：80000张，测试图片：20000张；</br>
图片形式如下： [验证码示例]</br>
![6JNTQe.png](https://i.loli.net/2018/10/11/5bbeb878b394a.png)![0jIW.png](https://i.loli.net/2018/10/11/5bbeb8fa3ec24.png)
#### 2.gen_tfrecords.py 将验证码识别模型的图片数据生成tfrecord格式，数据包括图片数据224*224，标签数据0-9，a-z，A-Z</br>
TFRecords是一种二进制文件，能更好的利用内存，更方便复制和移动，并且不需要单独的标签文件。</br>
生成tfrecord文件如下：6个字符验证码识别模型所用图片数据：train_6label.tfrecords，test_6label.tfrecords；四个字符验证码识别模型所用图片数据：train_4label.tfrecords，test_4label.tfrecords</br>
#### 3.gen_num_label_tf.py 识别验证码字符个数模型所需要的图片数据，转化为tfrecord格式，数据包括图片数据224*224，标签数据：0或1，0代表4个字符，1代表6个字符</br>
生成tfrecord文件如下：train.tfrecords，test.tfrecords</br>
#### 4.train_label_num.py 训练字符个数识别模型</br>
图片输入为[batch_size,weight,height],标签数据转为one-hot格式，如[1]=>[0,1],[0]=>[1,0]，输出端采用softmax函数，softmax模型可以用来给不同的对象分配概率。代价函数采用与softmax对应的对数似然函数，优化器采用速度快，效果好，可以自动调整学习率的AdadeltaOptimizer，初始步长为0.01，每迭代5000次降为原来的一半。网络结构为：第一个卷基层：卷积核5*5，步长为1，same方式，输出特征个数32；池化层：核4*4，步长4*4，same方式；第二个卷积层：核5*5，步长为1，same方式，输出特征个数64；池化层：核4*4，步长4*4，same方式；全连接层：w1 = [16*16*64，1024]，b1 = 1024;输出层w2 = [1024,2],b2 = [2]。为了防止过拟合，加入正则化，并令keep_prob=0.6，训练80000次，正确率达97.73%。</br>
#### 5.train_4_labels 4位数验证码图片识别</br>
网络结构自于Alex在2012年发表的经典论文AlexNet，conv1阶段DFD（data flow diagram）：![20170516212848372.png](https://i.loli.net/2018/10/11/5bbec61e7f9b0.png)
conv2阶段DFD（data flow diagram）：![20170516212902357.png](https://i.loli.net/2018/10/11/5bbec66a9c9a7.png)
conv3阶段DFD（data flow diagram）：![20170516212921278.png](https://i.loli.net/2018/10/11/5bbec68fa51b7.png)
conv4阶段DFD（data flow diagram）：![20170516213035624.png](https://i.loli.net/2018/10/11/5bbec6be8417e.png)
conv5阶段DFD（data flow diagram）：![20170516213050279.png](https://i.loli.net/2018/10/11/5bbec6f1f1be0.png)
fc6阶段DFD（data flow diagram）：![20170516213056139.png](https://i.loli.net/2018/10/11/5bbec73004a20.png)
fc7阶段DFD（data flow diagram）：![20170516213102123.png](https://i.loli.net/2018/10/11/5bbec7451de83.png)
fc8阶段DFD（data flow diagram）：![20170516213108672.png](https://i.loli.net/2018/10/11/5bbec75682e5b.png)</br>
具体原理参考：(https://blog.csdn.net/zyqdragon/article/details/72353420)</br>
训练次数为50000次，还没跑完，结果未知。</br>
#### 6.train_6_labels.py 6位数验证码识别</br>
网络结果同样使用AlexNet，输出个数改为6个，程序同样还没跑完，等待结果。