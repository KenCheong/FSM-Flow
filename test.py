import os
import matplotlib.pyplot as plt

import pandas as pd
import json
import time
from  sythetic_data_generator import *
from PSM_Flow import *
def load_result(fname):
    with open (fname, 'rb') as fp:
        matrix,labels,motif_assign_vec = json.load(fp)
    return matrix,labels,motif_assign_vec

def matrix_to_matrices(matrix,labels):
    components=get_connected_components(matrix,labels)
    doc_labels=[0]*len(labels)

    for c in range(len(components)):
        for i in range(len(components[c])):
#            print len(doc_labels),components[c][i]
            doc_labels[components[c][i]]=c

    doc_num=len(set(doc_labels))
    #doc_num=len(components)
    doc_components=[]
    matrices=[]
    label_list=[]
    for i in range(doc_num):doc_components.append([])
    for i in range(len(doc_labels)):doc_components[doc_labels[i]].append(i)
    for i in range(doc_num):
        component_matrix,component_labels=extract_matrix(matrix,labels,doc_components[i])
        matrices.append(component_matrix)
        label_list.append(component_labels)
    return matrices,label_list


f_psm=open('psm_stat.csv','a')
for i in range(10,80,10):
    matrix,labels,subgraphs_matrices,subgraphs_labels,ref_matrix,ref_labels= generate_sythetic_data(20,workflow_num=i,frag_size=15)
    matrices,label_list=matrix_to_matrices(matrix,labels)
    f_psm=open('psm_stat.csv','a')
    g,lp=matrix_to_graph(matrix,labels)
    int_labels,ind_label,index_label_dict=get_int_labels(labels)##map node labels of nodes to integer 
    dis_matrices,components=get_distance_matrices(matrix,labels)
    start = time.time()
    topic_assign_vec,pos_vectors,frag_means,frag_group,frag_topic,transition_matrices,perplexity_list=PSM_Flow(matrix,int_labels,dis_matrices,components,1,1,beta=0.5,dimensionlen=5,ind_label=ind_label,tau=2)
    end = time.time()
    elapsed = end - start
    print 'psm',elapsed
    f_psm.write('{0},{1}\n'.format(i,elapsed))
    f_psm.close()


##plot runtime
'''
df = pd.read_csv('psm_stat2.csv')  
print df.iloc[:,0].tolist()
print df.iloc[:,1].tolist()
x=df.iloc[:,0].tolist()
y=df.iloc[:,1].tolist()


plt.xlabel('Number of Transactions ')
plt.ylabel('Runtime(sec)')
plt.plot(x,y,marker='o')

plt.show()


'''
