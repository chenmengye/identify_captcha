# coding: utf-8
import random
import os
from captcha.image import ImageCaptcha, random_color
from contextlib import closing
from multiprocessing import Pool

__author__ = 'cmy'
char_set = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
img_dir = None
num_process = 12

def gen_char(char_num):
    #生成char_num位字符
    buf = ""
    for i in range(char_num):
        buf += random.choice(char_set)
    return buf

def generate_img(char_num):
    global img_dir
    the_chars = gen_char(char_num)
    # 选择验证码的字体，图片大小，字体大小
    captcha = ImageCaptcha(fonts=['./font/simhei.ttf'])
    # 设置背景，颜色
    backgroud = random_color(238, 255)
    color = random_color(0, 200, opacity=None)
    # 创建图片
    img = captcha.create_captcha_image(the_chars, color, backgroud)
    captcha.create_noise_curve(img, color)
    captcha.create_noise_dots(img, color, number=None)
    from PIL import ImageFilter
    img.filter(ImageFilter.SMOOTH)
    # 保存生成的验证码，按照：序号_结果.png 保存
    img_name = the_chars + '.png'
    img_path = img_dir + '/' + img_name
    captcha.write(the_chars, img_path)
    # print img_path

def run(num, path,char_num):
    global img_dir
    img_dir = path
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(num):
        print i
        generate_img(char_num)


if __name__ == '__main__':
    # 用来生成训练模型用的验证码
    run(80000, 'train_imgs_4',4)
    run(80000, 'train_imgs_6',6)
    run(20000, 'test_imgs_4',4)
    run(20000, 'test_imgs_6',6)