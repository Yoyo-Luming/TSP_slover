import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

from utils import *


class DP():
    def __init__(self, num_node, data):
        self.num_city = num_node
        self.location = data
        self.dis_mat = compute_dis_matrix(num_node, data)
        self.m = 1 << num_node
        self.dp = np.zeros((num_node, self.m))

    def run(self):
        ans = self.find(0, [x for x in range(1, self.num_city)])
        lists = [x for x in range(0, self.num_city)]
        start = 0
        path = [[self.location[0][0], self.location[0][1]]]
        while len(lists) > 0:
            lists.pop(lists.index(start))
            idx = self.transfer(lists)
            next_node = self.dp[start][idx]
            start = int(next_node)
            path.append([self.location[start][0],self.location[start][1]])
        return np.array(path), ans

    def transfer(self, sets):
        idx = 0
        for s in sets:
            idx = idx + 2 ** s  # 二进制转换
        return idx

    def find(self, node, future_sets):
        if len(future_sets) == 0:
            return self.dis_mat[node][0]
        d = 99999
        distance = []
        for i in range(len(future_sets)):
            s_i = future_sets[i]
            list_copy = future_sets[:]
            # print(i, list_copy, future_sets)
            list_copy.pop(i)  # 删除第i个节点，认为已经完成对其的访问
            distance.append(self.dis_mat[node][s_i] + self.find(s_i, list_copy))
            # 动态规划递推方程，利用递归
        d = min(distance)
        # node需要连接的下一个节点
        next_one = future_sets[distance.index(d)]
        # 未遍历节点集合
        c = self.transfer(future_sets)
        # 回溯矩阵，（当前节点，未遍历节点集合）——>下一个节点
        self.dp[node][c] = next_one
        return d


def main():
    # data = read_tsp("../data/ulysses16.tsp")
    # data = np.array(data)
    # data = data[:, 1:]
    data = [[38.24, 20.42], [39.57, 26.15], [38.15,15.35],[40.56,25.32],[37.51,15.17],[35.49,14.32],[39.36,19.56],
            [36.26,23.12], [33.48, 10.54],[38.42, 13.11], [37.52, 20.44]]
    data = np.array(data)

    matplotlib.rc("font", family='Microsoft YaHei')
    model = DP(num_node=data.shape[0], data=data.copy())
    Best_path, Best = model.run()
    print('规划的路径长度:{}'.format(Best))
    # 显示规划结果
    plt.scatter(Best_path[:, 0], Best_path[:, 1])
    Best_path = np.vstack([Best_path, Best_path[0]])
    plt.plot(Best_path[:, 0], Best_path[:, 1])
    plt.title('规划路线结果')
    plt.show()


if __name__ == '__main__':
    main()
