import numpy as np


def compute_dis_matrix(num_node, location):
    dis_matrix = np.zeros((num_node, num_node))
    for i in range(num_node):
        for j in range(num_node):
            if i == j:
                dis_matrix[i][j] = np.inf
                continue
            a = location[i]
            b = location[j]
            dis_matrix[i][j] = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
    return dis_matrix


def compute_pathlen(path, dis_mat, goback=True):
    try:
        a = path[0]
        b = path[-1]
    except:
        import pdb
        pdb.set_trace()
    if goback:
        result = dis_mat[a][b]
    else:
        result = 0.0
    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        result += dis_mat[a][b]
    return result

def read_tsp(path):
    lines = open(path, 'r').readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    return data