# different_words_captcha
identify four_chars captcha or six_chars captcha
</br>
### <font color=#00FFFF >���ȼ򵥽���һ�������˼·,����Ҫʶ�����֤����4λ��6λ��������ѵ��һ��ģ����ʶ����֤����ַ��������پ�������ʶ����ȷ����4λ��ģ�ͻ���6λ��ģ��</font>
#### 1.gen_img.py ������֤��ͼƬ</br>
��λ����֤��ѵ��ͼƬ��80000�ţ�����ͼƬ��20000�ţ���λ����֤��ѵ��ͼƬ��80000�ţ�����ͼƬ��20000�ţ�</br>
ͼƬ��ʽ���£� [��֤��ʾ��]</br>
![6JNTQe.png](https://i.loli.net/2018/10/11/5bbeb878b394a.png)![0jIW.png](https://i.loli.net/2018/10/11/5bbeb8fa3ec24.png)
#### 2.gen_tfrecords.py ����֤��ʶ��ģ�͵�ͼƬ��������tfrecord��ʽ�����ݰ���ͼƬ����224*224����ǩ����0-9��a-z��A-Z</br>
TFRecords��һ�ֶ������ļ����ܸ��õ������ڴ棬�����㸴�ƺ��ƶ������Ҳ���Ҫ�����ı�ǩ�ļ���</br>
����tfrecord�ļ����£�6���ַ���֤��ʶ��ģ������ͼƬ���ݣ�train_6label.tfrecords��test_6label.tfrecords���ĸ��ַ���֤��ʶ��ģ������ͼƬ���ݣ�train_4label.tfrecords��test_4label.tfrecords</br>
#### 3.gen_num_label_tf.py ʶ����֤���ַ�����ģ������Ҫ��ͼƬ���ݣ�ת��Ϊtfrecord��ʽ�����ݰ���ͼƬ����224*224����ǩ���ݣ�0��1��0����4���ַ���1����6���ַ�</br>
����tfrecord�ļ����£�train.tfrecords��test.tfrecords</br>
#### 4.train_label_num.py ѵ���ַ�����ʶ��ģ��</br>
ͼƬ����Ϊ[batch_size,weight,height],��ǩ����תΪone-hot��ʽ����[1]=>[0,1],[0]=>[1,0]������˲���softmax������softmaxģ�Ϳ�����������ͬ�Ķ��������ʡ����ۺ���������softmax��Ӧ�Ķ�����Ȼ�������Ż��������ٶȿ죬Ч���ã������Զ�����ѧϰ�ʵ�AdadeltaOptimizer����ʼ����Ϊ0.01��ÿ����5000�ν�Ϊԭ����һ�롣����ṹΪ����һ������㣺�����5*5������Ϊ1��same��ʽ�������������32���ػ��㣺��4*4������4*4��same��ʽ���ڶ�������㣺��5*5������Ϊ1��same��ʽ�������������64���ػ��㣺��4*4������4*4��same��ʽ��ȫ���Ӳ㣺w1 = [16*16*64��1024]��b1 = 1024;�����w2 = [1024,2],b2 = [2]��Ϊ�˷�ֹ����ϣ��������򻯣�����keep_prob=0.6��ѵ��80000�Σ���ȷ�ʴ�97.73%��</br>
#### 5.train_4_labels 4λ����֤��ͼƬʶ��</br>
����ṹ����Alex��2012�귢��ľ�������AlexNet��conv1�׶�DFD��data flow diagram����![20170516212848372.png](https://i.loli.net/2018/10/11/5bbec61e7f9b0.png)
conv2�׶�DFD��data flow diagram����![20170516212902357.png](https://i.loli.net/2018/10/11/5bbec66a9c9a7.png)
conv3�׶�DFD��data flow diagram����![20170516212921278.png](https://i.loli.net/2018/10/11/5bbec68fa51b7.png)
conv4�׶�DFD��data flow diagram����![20170516213035624.png](https://i.loli.net/2018/10/11/5bbec6be8417e.png)
conv5�׶�DFD��data flow diagram����![20170516213050279.png](https://i.loli.net/2018/10/11/5bbec6f1f1be0.png)
fc6�׶�DFD��data flow diagram����![20170516213056139.png](https://i.loli.net/2018/10/11/5bbec73004a20.png)
fc7�׶�DFD��data flow diagram����![20170516213102123.png](https://i.loli.net/2018/10/11/5bbec7451de83.png)
fc8�׶�DFD��data flow diagram����![20170516213108672.png](https://i.loli.net/2018/10/11/5bbec75682e5b.png)</br>
����ԭ��ο���(https://blog.csdn.net/zyqdragon/article/details/72353420)</br>
ѵ������Ϊ50000�Σ���û���꣬���δ֪��</br>
#### 6.train_6_labels.py 6λ����֤��ʶ��</br>
������ͬ��ʹ��AlexNet�����������Ϊ6��������ͬ����û���꣬�ȴ������