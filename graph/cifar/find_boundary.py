# -*- coding: utf-8 -*-
# @Author  : lan
# @Software: PyCharm
import networkx as nx
# import skimage.future.graph as graph
import skimage.graph as graph
import cv2
import itertools
import scipy.sparse as sp

from PIL import Image, ImageDraw
from skimage.segmentation import slic, mark_boundaries
from matplotlib import pyplot as plt
from skimage import io, segmentation, util, measure, color, transform

import numpy as np
from skimage.segmentation import slic, find_boundaries
from skimage.color import rgb2lab
from skimage import io
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray


def cut_threshold(labels, rag, thresh, in_place=True):
    if not in_place:
        rag = rag.copy()

    # Because deleting edges while iterating through them produces an error.
    to_remove = [(x, y) for x, y, d in rag.edges(data=True)
                 if d['weight'] >= thresh]
    rag.remove_edges_from(to_remove)

    comps = nx.connected_components(rag)

    # We construct an array which can map old labels to the new ones.
    # All the labels within a connected component are assigned to a single
    # label in the output.
    map_array = np.arange(labels.max() + 1, dtype=labels.dtype)
    for i, nodes in enumerate(comps):
        for node in nodes:
            for label in rag.nodes[node]['labels']:
                map_array[label] = i + 1

    return map_array[labels]

def get_boundary(img, K=100, M=10): # 200  150

    image = cv2.resize(img, (224, 224))
    # Convert RGB image to LAB color space
    lab = rgb2lab(image)

    segments = slic(lab, n_segments=K, compactness=M, sigma=5, start_label=1)  # ,start_label=1
    # Merge similar areas, there is its own RAG map here
    rag = graph.rag_mean_color(image, segments)

    segments = cut_threshold(segments, rag, 5)

    # Get the label matrix of the region
    labels = measure.label(segments, connectivity=2)

    pixel_nums = np.bincount(labels.flat)[1:]

    # Calculate the center point position of each superpixel block
    regions = measure.regionprops(labels)

    feature_List = []
    gray_img = util.img_as_ubyte(color.rgb2gray(image))

    for i in range(len(regions)):
        mask = labels == i + 1

        cen = regions[i]

        centroid = cen.centroid 
        local_cen = cen.local_centroid

        area = round(cen.area, 4)

        perimeter = round(cen.perimeter, 4) 

        # moments_central = cen.moments_central.flatten().tolist() 
        avg = round(np.mean(image[mask]), 4) 

        max = round(np.max(image[mask]), 4)

        min = round(np.min(image[mask]), 4)

        var = round(np.var(image[mask]), 4)

        # Get the coordinate range of the region
        min_row, min_col, max_row, max_col = cen.bbox  # Bounding box of region block(min_row, min_col, max_row, max_col)
        min_row, min_col, max_row, max_col = round(min_row, 4), round(min_col, 4), round(max_row, 4), round(max_col, 4)

        feature_List.append([
                                round(centroid[0]), round(centroid[1]), round(local_cen[0]), round(local_cen[1]), area,
                                perimeter, avg, max, min, var,
                                min_row, min_col, max_row, max_col])

    List_node = []
   
    label_pos = {i: [] for i in range(1, len(pixel_nums) + 1)}
    count = 1
    pos_dict = {}  
    node_counts = []
    for i in range(0, 224, 16):  
        for j in range(0, 224, 16):
            pos_dict[count] = [(i + i + 16) // 2, (j + j + 16) // 2]

            label_patch = labels[i:i + 16, j:j + 16] 

            flat_labels = [label for row in label_patch for label in row]
            # Count the number of each label
            label_counts = {}
            for label in flat_labels:
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
            node_counts.append(label_counts)

            total = sum(label_counts.values())  

            temp = np.zeros(14) # 14个特征
            # Traverse each key-value pair, calculate the weight ratio of the value, and output the result
            for k, v in label_counts.items():
                weight = v / total  # Calculate the weight proportion of this value
                temp += weight * np.array(feature_List[k - 1])  # Pixel markers are 1-97（because measure.regionprops(labels) want to set 0 as the background. , note that the subscripts start with 0）

                # Add the corresponding node to each tag
                label_pos[k].append(count)

            List_node.append(temp)
            count += 1


    # According to label_pos, traverse the node where each superpixel block is located, and add edges between these nodes
    List_i = []
    List_y = []
    G = nx.Graph()
    for i in range(1, 197): 
        G.add_node(i)
    

    all_res = []
    all_weight = []

    theLine = 0
    for pix_label, nodes in label_pos.items():

        res = list(itertools.combinations(nodes, 2))

        for kk in range(len(res)):
        
            weight = node_counts[res[kk][0]-1][pix_label]/pixel_nums[pix_label-1]+node_counts[res[kk][1]-1][pix_label]/pixel_nums[pix_label-1]
            
            if weight >= 0:
                all_res.append((res[kk][0],res[kk][1],weight))
                all_weight.append(weight)
                theLine += 1

    G.add_weighted_edges_from(all_res)  # Connect a line to each of the permutations
    print('theLine--------------',theLine)
    # The node number of G is 1-197, and the connection of adj side also starts with 1.
    # print('G', G)

    # print('pos_dict', pos_dict)
    # nx.draw_networkx_nodes(G, pos_dict, node_size=1, node_color='black')  
    # nx.draw_networkx_edges(G, pos_dict, width=0.5, alpha=1, edge_color='blue')  

    a = nx.to_numpy_matrix(G)  
    adj = a.A  
    adj = sp.coo_matrix(adj) 
    adj = np.vstack((adj.row+1, adj.col+1)) 
    edge_attr = np.zeros((len(adj[0]), 2))
    for i in range(len(adj[0])):
        a = pos_dict[adj[0][i]]
        b = pos_dict[adj[1][i]]
        dis = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
        wei = G.get_edge_data(adj[0][i], adj[1][i])['weight']
        edge_attr[i] = [dis, wei] 


    return List_node, adj, edge_attr


if __name__ == '__main__':
    img = cv2.imread('../test.jpg')
    get_boundary(img)
