#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2020/7/31 上午11:54
@Author : Catherinexxx
@Site : 
@File : test.py.py
@Software: PyCharm
"""
from bert_utils.classifier_args import classifier_args

seq_args = classifier_args.copy()
print(seq_args['num_epochs'])