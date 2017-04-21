from nutils import *
import numpy as np
import utilities as ut
from auxilliary_classes import *
import test_bench as tb
from matplotlib import pyplot as plt

ndims1, ndims2 = [6,10], [22,13]

degree = 2

go1 = tb.square_go(ndims1, degree)
go2 = tb.circle_go(ndims2, degree)

(go1.ref_by([[2,4], [4,5]]) + go2).quick_plot()
(go1 - go2.ref_by([[1,2], [12,11]])).quick_plot()
(go2.ref_by([[2,6], [4,5]]) % go1).quick_plot()
(go2 | go1.ref_by([[1,2], [3,6]])).quick_plot()

side = 'bottom'

(go2[side] + go1[side]).ref_by([[5,6]]).quick_plot()
(go2[side] - go1[side]).ref_by([[5,6]]).quick_plot()
(go2[side] | go1[side]).ref_by([[5,6]]).quick_plot()
(go2[side] % go1[side]).ref_by([[5,6]]).quick_plot()