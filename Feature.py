# Rutger Introduction To Artificial Intelligence (CS 520) Final project
# Feature.py
# Date: 04/08/2017
# Created by Kaixiang Huang, Yehan Wang
# Based on the  http://inst.eecs.berkeley.edu/~cs188/sp11/projects/classification/Feature.py
# Artificial Intelligence - Berkeley cs188

import numpy as np


def basicFeaturesExtract(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    for both Face image and digit image
    """
    features = []
    for x in range(datum.height):
        row = []
        for y in range(datum.width):
            if datum.getPixel(x, y) > 0:
                row.append(1)
            else:
                row.append(0)
        features.append(row)
    return np.array(features)



def in_boundary(x, y, datum):
    # to find whether this pixels position is in or out of boundary
    width = datum.width
    height = datum.height
    if x >= 0 and y >= 0 and  x < height + 2 and y < width + 2:
        return True
    else:
        return False



def find_neighbors(x, y, datum, visited):
    # find position's available neighbor
    neighbor = []
    if in_boundary(x-1, y, datum) and visited[x-1][y] != 1:
        neighbor.append((x-1, y))
    if in_boundary(x, y-1, datum) and visited[x][y-1] != 1:
        neighbor.append((x, y-1))
    if in_boundary(x+1, y, datum) and visited[x+1][y] != 1:
        neighbor.append((x+1, y))
    if in_boundary(x, y+1, datum) and visited[x][y+1] != 1:
        neighbor.append((x, y+1))
    return neighbor



def cycle_finder(datum):
    """
    find cycle feature of image by DFS, if we can find some zero in image map which can not be searched by DFS, there may
    be a cycle that subtended by 1, but there are still some mistakes that happened because of image's boundary. So I use
    zero_padding in image before DFS to get better result
    """
    feature = basicFeaturesExtract(datum)
    visited = np.array(feature)
    visited = np.lib.pad(visited, 1, 'constant')
    feature_p = visited.tolist()
    cycle_num = -1
    cycle_map = np.zeros((1, datum.width))

    while np.count_nonzero(visited) != visited.size:
        open_list = []
        open_list.append(np.unravel_index(visited.argmin(), visited.shape))
        current = open_list[0]

        # kick out some small cycles which is only one pixel
        initial_neighbor = find_neighbors(current[0], current[1], datum, visited)
        if initial_neighbor==[]:
            visited[current[0]][current[1]] = 1
            continue

        while len(open_list):
            current = open_list.pop()
            visited[current[0]][current[1]] = 1
            neighbor = find_neighbors(current[0], current[1], datum, visited)
            for i in neighbor:
                if feature_p[i[0]][i[1]] != 1:
                    open_list.append(i)
        cycle_num += 1

    for i in range(cycle_num):
        cycle_map[0][i] = 1
    feature = np.vstack((feature, cycle_map))
    return feature


def flatten_feature(image_data, cycle):
    """
    flatten input data's feature and prepare it to feed the neural network
    """
    flatten_features = np.array([])
    for i in range(len(image_data)):
        if cycle == 0:
            feature = basicFeaturesExtract(image_data[i])
        elif cycle == 1:
            feature = cycle_finder(image_data[i])
        flatten_features = np.append(flatten_features, feature.flatten(), axis=0)
    flatten_features = np.reshape(flatten_features, (len(image_data), feature.size))
    return flatten_features
