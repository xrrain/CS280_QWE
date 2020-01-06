import networkit as nk
import os
import os.path as path
import time
root_path = "data/DrBC_Data/Data"
second = ["Real", "Synthetic"]
real_file_path = path.join(root_path, second[0])
real_file = os.listdir(real_file_path)
graphs = {}
for file in real_file:
    graph_path = path.join(real_file_path, file)
    if path.isdir(graph_path):
        continue
    graph_name = file.split(".")[0]
    graphs[graph_name] = nk.readGraph(graph_path, nk.Format.EdgeList, separator=" ", firstNode=0)
    print("finish %s" % graph_name)
    

for graph_name, graph in graphs.items():
    root_dir = path.join(real_file_path, "estimated_bc")
    result_file = "%s.txt" % graph_name
    start = time.time()
    res = nk.centrality.ApproxBetweenness(graph)
    end = time.time()
    print(type(res))
    break