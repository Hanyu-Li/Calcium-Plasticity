from graph_tool.all import *
import numpy as np
import pylab as plt
g, pos = triangulation(np.random.random((500, 2)) * 4, type="delaunay")
tree = min_spanning_tree(g)
graph_draw(g, pos=pos, edge_color=tree, output="min_tree.pdf")
g.set_edge_filter(tree)
graph_draw(g, pos=pos, output="min_tree_filtered.pdf")
