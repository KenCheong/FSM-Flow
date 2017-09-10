from toy_graph_generater import *
from utility import *
from PSM_Flow_DE import *
from PSM_Flow_GC import *
from postprocess import *

topic_num=3
matrix,labels,subgraphs_matrices,subgraphs_labels=get_toy_graph(repeat_times=10)
g,lp=matrix_to_graph(matrix,labels)
int_labels=get_int_labels(labels)##map node labels of nodes to integer 

###Distance encoded Gibb sampling
d_m=get_distance_mat(g,labels)
topic_assign_vec=DE_gibb_sampling(int_labels,d_m,topic_num,100,verbose=True)
###Gaussian cluster LDA
'''
node_vectors,components=get_node_vectors(matrix,labels)
topic_assign_vec=GC_LDA(int_labels,node_vectors,components,topic_num,50)
'''

###output visualization
plot_graph(matrix,labels,topic_assign_vec,fname='toy_graph')
frag_matrix,frag_labels,topic_assign_vec=extract_fragments(matrix,labels,topic_assign_vec)#get fragments
plot_graph(frag_matrix,frag_labels,topic_assign_vec,fname='fragments')


