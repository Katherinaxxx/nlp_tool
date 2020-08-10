#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2020/7/29 下午3:00
@Author : Catherinexxx
@Site : 
@File : word_frequency.py
@Software: PyCharm
"""


import os, sys
import re
import matplotlib.pyplot as plt
import pandas as pd
import jieba
import jieba.posseg as pseg
from collections import Counter
jieba.enable_paddle()
import seaborn as sns
from wordcloud import WordCloud


def common_word(txt_file1, txt_file2):
    df1 = pd.read_csv(txt_file1, sep=':')
    df2 = pd.read_csv(txt_file2, sep=':')
    df1.columns = ['word','count']
    df2.columns = ['word','count']
    df = pd.merge(df1, df2, on=['word'])
    draw_histogram(df.iloc[:,1], df.iloc[:,0], txt_file1+'.png')
    draw_histogram(df.iloc[:,2], df.iloc[:,0], txt_file2+'.png')


def txt_frequency(txt_file):
    words = open(txt_file, 'r', encoding='utf-8')
    freq = {}
    for line in words.readlines():
        word = line.split(':')
        freq[word[0]] = int(word[1])
    return freq


def draw_wordcloud(txt_file):
    freq, _ = txt_frequency(txt_file)
    wc = WordCloud(font_path='./font/msyh.ttc',  # 设置字体
                   background_color="white",  # 背景颜色
                   max_words=2000,  # 词云显示的最大词数
                   max_font_size=90,  # 字体最大值
                   random_state=41,
                   scale=3
                   )
    wc.generate_from_frequencies(freq)
    wc.to_file(txt_file+'.png')


def draw_histogram(data, label, fig_path):
    sns.set_style('darkgrid')
    ax = sns.distplot(x='x', y='y', data=data, hist=True)
    # ax = sns.distplot(data, label=label, hist=True)
    hist_fig = ax.get_figure()
    hist_fig.savefig(fig_path)


def segment_words(sen):
    '''
    segment words with jieba
    '''
    words, tags = [], []
    m = pseg.cut(sen, use_paddle=True)
    for x in m:
        words.append(x.word)
        tags.append(x.flag)
    return words


def word_frequency(file_name):
    all_words = []
    with open(file_name, 'r', encoding='utf-8') as t:
        lines = t.readlines()
        for line in lines:
            cut_words = segment_words(line)
            all_words.extend(cut_words) # 将分词结果用set过滤，同一个词语在同一首歌内仅统计一次

    count = Counter(all_words) # 计算次数
    sorted_count = sorted(count.items(), key=lambda x: x[1], reverse=True) # 排序统计结果

    # 保存结果
    fw = open('more_than_two_words_'+file_name, 'w', encoding='utf-8')
    for item in sorted_count:
        if len(item[0])>1:
            fw.write(item[0] + ':' + str(item[1]) + '\n')
    fw.close()


if __name__ == '__main__':
    # file_name = None
    # word_frequency(file_name)
    # txt_file = '/Users/xyh/Downloads/mayday_lyric_analyze-master/cloud.txt'
    # draw_wordcloud(txt_file)
    # draw_histogram(txt_file)
    common_word('/Users/xyh/Downloads/mayday_lyric_analyze-master/cloud.txt', '/Users/xyh/Downloads/mayday_lyric_analyze-master/cloud2.txt')