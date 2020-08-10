#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2020/7/29 下午4:09
@Author : Catherinexxx
@Site : 
@File : split_by_word.py
@Software: PyCharm
"""
def split_by_word(s):
    temp = ''
    lst = []
    for c in s:
        if '\u4e00' <= c <= '\u9fff':
            if temp != '':
                lst.append(temp)
                temp = ''
            lst.append(c)
        else:
            temp += c
    return lst
