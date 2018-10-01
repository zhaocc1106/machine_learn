#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
The Convolution Neural Networks Algorithm.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/9/29 19:48
"""

# Third-party library
import theano
import theano.tensor as T
from theano import function
import numpy as np


#### Constants
# GPU = True
# if GPU:
#     print "Trying to run under a GPU.  If this is not desired, then modify "+\
#         "network3.py\nto set the GPU flag to False."
#     try: theano.config.device = 'gpu'
#     except:
#         print("exception")
#     theano.config.floatX = 'float32'
# else:
#     print "Running with a CPU.  If this is not desired, then the modify "+\
#         "network3.py to set\nthe GPU flag to True."


if __name__=="__main__":
    x = T.dscalar('x')
    y = T.dscalar('y')
    z = x + y
    f = function([x, y], z)
    print f(2, 3)
    print("openmp:" + str(theano.config.openmp))
    print("device:" + str(theano.config.device))
    print("floatX:" + str(theano.config.floatX))
