import copy
from random import randint,uniform,shuffle
from utility import *


def is_connected(subgraph_matrix):
    node_num=len(subgraph_matrix)
   # print node_num
    subgraph_matrix=copy.deepcopy(subgraph_matrix)
    for i in range(node_num):
        for j in range(node_num):
            if subgraph_matrix[i][j]==1:
                subgraph_matrix[j][i]=1
    node_list=[0]
    visit_nodes=set()
    while(len(node_list)>0):
        for i in range(node_num):
            if i not in visit_nodes and subgraph_matrix[node_list[0]][i]==1:
                node_list.append(i)
        visit_nodes.add(node_list[0])
        node_list=node_list[1:]
    return len(visit_nodes)==node_num

def generate_subgraph(subgraph_matrix):
    node_num=len(subgraph_matrix)
    while(is_connected(subgraph_matrix)==False):
        i=randint(0,node_num-1)
        j=randint(0,node_num-1)
        if i==j:continue
        subgraph_matrix[i][j]=1
#    for i in range(node_num-1):
 #       subgraph_matrix[i][i+1]=1
    return subgraph_matrix


def generate_sythetic_data(topic_num,label_num=100,frag_size=10,workflow_num=40,frag_per_workflow=5,repeat_times=10,start_sym=0):
    ##subgraph preparation
    topic_labels=[]                        
    for i in range(topic_num):topic_labels.append([])
    topic_symbols=[]
    for i in range(label_num):
        topic_symbols.append('A'+str(start_sym)+str(i))
    for t in range(topic_num):
        for i in range(frag_size):
            pick_label=randint(0,label_num-1)
            topic_labels[t].append(topic_symbols[pick_label])
    total_topic_labels=[]
    for i in range(topic_num):
        total_topic_labels+=topic_labels[i]
    index_label_dict={}
    for ind,la in enumerate(total_topic_labels):
        index_label_dict[la]=ind
    empty_net=[]
    for i in range(topic_num):
        empty_net.append([])
    for i in range(topic_num):
        for j in range(len(topic_labels[i])):
            empty_net[i].append([0]*len(topic_labels[i])) 

    topic_structure=[]
    for t in range(topic_num):
        topic_structure.append(copy.deepcopy(empty_net[t]))

    for t in range(topic_num):
        topic_structure[t]=generate_subgraph(topic_structure[t])
  
    reference_subgraph_matrix,_= matrices_to_one_matrix(topic_structure,[])

    ##
    subgraphs_matrices,subgraphs_labels=[],[]
    for i in range(workflow_num):
        tms,tls=[],[]
        for j in range(frag_per_workflow):
            pick_topic=randint(0,topic_num)
            tms.append(topic_structure[j])
            tls.append(topic_labels[j])
        tmatrix,tlabels= matrices_to_one_matrix(tms,tls)
        tmatrix=generate_subgraph(tmatrix)
        subgraphs_matrices.append(tmatrix)
        subgraphs_labels.append(tlabels)
    matrix,labels= matrices_to_one_matrix(subgraphs_matrices,subgraphs_labels)

    return matrix,labels,subgraphs_matrices,subgraphs_labels,reference_subgraph_matrix,total_topic_labels
