# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:02:58 2016

@author: Ilho
"""


import pandas
from IPython.display import display

data = pandas.read_csv('stroopdata.csv')
display(data.head())