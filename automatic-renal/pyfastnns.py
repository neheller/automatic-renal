#! /usr/bin/python
# -*- coding: utf-8 -*-
u"""
Fast Nearest Neighbor Search on python using kd-tree

author Atsushi Sakai

usage: see test codes as below

license: MIT
"""
import numpy as np
import scipy.spatial


class NNS:

    def __init__(self, data):
        # store kd-tree
        self.tree = scipy.spatial.cKDTree(data)

    def search(self, inp):
        u"""
        Search NN

        inp: input data, single frame or multi frame

        """

        if len(inp.shape) >= 2:  # multi input
            index = []
            dist = []

            # print(inp.shape)
            # print(inp.T.shape)

            for i in inp:
                # print(i)
                idist, iindex = self.tree.query(i)
                index.append(iindex)
                dist.append(idist)

            return index, dist
        else:
            dist, index = self.tree.query(inp)
            return index, dist

    def search_in_distance(self, inp, r):
        u"""
        find points with in a distance r
        """

        index = self.tree.query_ball_point(inp, r)
        return index


def test_2d():
    import matplotlib.pyplot as plt
    data2d = np.random.random(10000).reshape(5000, 2)
    print(data2d)

    #  input2d = np.random.random(2).reshape(2, 1)
    input2d = np.random.random(2)
    print(input2d)

    nns1 = NNS(data2d)

    index, dist = nns1.search(input2d)
    print(index, dist)

    #  data2d =

    plt.plot(data2d[:, 0], data2d[:, 1], ".r")
    plt.plot(input2d[0], input2d[1], "xk")
    plt.plot(data2d[index, 0], data2d[index, 1], "xb")
    plt.show()


def test_3d():
    # 3d
    data3d = np.random.random(240000).reshape(80000, 3)
    print(data3d.shape)

    #  input2d = np.random.random(2).reshape(2, 1)
    input3d = np.random.random(240000).reshape(80000, 3)
    print(input3d.shape)

    nns2 = NNS(data3d)

    index, dist = nns2.search(input3d)
    print(index, dist)


def test():
    data2d = np.random.random(10000).reshape(5000, 2)
    print(data2d)

    #  input2d = np.random.random(2).reshape(2, 1)
    input2d = np.random.random(6).reshape(2, 3)
    print(input2d)

    nns = NNS(data2d)

    index, dist = nns.search(input2d)
    print(index, dist)


if __name__ == '__main__':
    # test()
    test_3d()