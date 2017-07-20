
from sklearn.manifold import MDS
from scipy.spatial import distance
import csv
import numpy as np
import json
import sys
import pickle
import copy
from graph_tool.all import *
from numpy import argmax
import lda
import graphviz as gv
from random import randint,uniform,shuffle
import numpy as np
import math 
import scipy as sc

 

def matrix_to_edges(matrix,labels):
    edges=[]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j]==1:
                edges.append((str(i),str(j)))
    return edges

def add_nodes(graph, nodes):
    for n in nodes:
        if isinstance(n, tuple):
            graph.node(n[0], **n[1])
        else:
            graph.node(n)
    return graph

def add_edges(graph, edges):
    for e in edges:
        if isinstance(e[0], tuple):
            graph.edge(*e[0], **e[1])
        else:
            graph.edge(*e)
    return graph
def get_doc_components(doc_labels):
    doc_num=len(set(doc_labels))
    doc_components=[]
    for i in range(doc_num):doc_components.append([])
    for i in range(len(doc_labels)):doc_components[doc_labels[i]].append(i)
    return doc_components
def plot_dir(matrix,labels,doc_labels,components=[],motif_assign_vec=[],dirname=''):
    if len(doc_labels)==0:#it is connected components!!!
        doc_labels=[0]*len(labels)
        if components==[]:
            components=get_connected_components(matrix,labels)

        for c in range(len(components)):
            for i in range(len(components[c])):
    #            print len(doc_labels),components[c][i]
                doc_labels[components[c][i]]=c
   # print components

    doc_num=len(set(doc_labels))
    #doc_num=len(components)
    print 'e',len(set(doc_labels)),max(doc_labels)
    doc_components=[]
    for i in range(doc_num):doc_components.append([])
    for i in range(len(doc_labels)):doc_components[doc_labels[i]].append(i)
    for i in range(doc_num):
        component_matrix,component_labels=extract_matrix(matrix,labels,doc_components[i])
        component_motif_assign_vec=[motif_assign_vec[l] for l in doc_components[i]]
        plot_graph2(component_matrix,component_labels,motif_assign_vec=component_motif_assign_vec,fname=dirname+str(i))
    return doc_components


def plot_graph2(matrix,labels,motif_assign_vec=[],fname='net_wc2_result'):#graphvz
    G = gv.Digraph(format='png')
    styles = {
    'graph': {
        'rankdir': 'random',
    }}
    G.graph_attr.update(('graph' in styles and styles['graph']) or {})


    color_map = {0: 'red',  1: 'blue',2: 'yellow',3:'grey',4:'pink',5:'cyan2',6:'azure',7:'bisque1',8:'gold',9:'blueviolet',10:'black',11:'navy',12:'ivory3',13:'magenta',14:'brown',15:'burlywood',16:'seagreen',17:'darkorange',18:'deeppink',19:'cyan4'}
    shape_map = {1:"box",0:"circle"}
    if len(motif_assign_vec)==0:
        add_nodes(G,[(str(i),{'label':labels[i]}) for i in range(len(labels))])
    else:

        add_nodes(G,[(str(i),{'label':labels[i],'color':color_map[motif_assign_vec[i]]}) for i in range(len(labels))])
    add_edges(G,matrix_to_edges(matrix,labels))
    G.render(fname)
def normalize(arr):
    s=sum(arr)
    for i in range(len(arr)):arr[i]/=float(s)
def sim(dis):
    var=100
    mean=0

    return math.exp(-(dis**2))

def get_motif_portion(node,node_group,distance_matrix,labels,motif_assign_vec,motif_num,ecount_matrix,motif_count,alpha=0.05,beta=0.05):
    label_num=len(set(labels))
    motif_portion=[0]*motif_num
    motif_neigh_portion=[0]*motif_num
    for i in range(motif_num):
        motif_portion[i]=(float(ecount_matrix[labels[node]][i])+alpha)/(motif_count[i]+alpha*label_num)
    normalize(motif_portion)
    #return motif_portion
    s=0.0
    #for i in range(len(labels)):
    for i in node_group[node]:
#        if i==node or distance_matrix[node][i]>4:continue
        s+=sim(distance_matrix[node][i])
        motif_neigh_portion[motif_assign_vec[i]]+=(sim(distance_matrix[node][i]))
   
    for i in range(motif_num):
        motif_neigh_portion[i]+=beta
    normalize(motif_neigh_portion)
    for i in range(motif_num):
        motif_portion[i]*=(motif_neigh_portion[i])
    normalize(motif_portion)
    return motif_portion

def motif_lda(labels,distance_matrix,w,node_num,motif_num,iter_num,verbose=False):
    motif_assign_vec=[0]*node_num
    motif_portion=[0]*motif_num
   
    ecount_matrix=np.zeros(shape=(w,motif_num),dtype=np.int)
    motif_count=np.zeros(shape=(motif_num))
    node_group=[]
    for ii in range(node_num):
        node_group.append([])
        for j in range(node_num):
            if ii!=j and distance_matrix[ii][j]<5:node_group[ii].append(j)
    #initialize
    for ii in range(node_num):
        picked_motif=np.random.choice(motif_num,1)[0]
        motif_assign_vec[ii]=picked_motif
#        print labels[ii]
        ecount_matrix[labels[ii]][picked_motif]+=1
        motif_count[picked_motif]+=1
    ######
    for i in range(iter_num): 
        print 'iter num:'+str(i)
        converge=True
        scan_order=range(node_num)
        shuffle(scan_order)
               
            #for e in range(len(group_set[group_id])):
        for node in scan_order:
            old_motif=motif_assign_vec[node]
            ecount_matrix[labels[node]][old_motif]-=1
            motif_count[old_motif]-=1

            motif_portion=get_motif_portion(node,node_group,distance_matrix,labels,motif_assign_vec,motif_num,ecount_matrix,motif_count)
           # print motif_portion
            picked_motif=np.random.choice(motif_num,1,p=motif_portion)[0]
#                picked_motif=np.random.choice(motif_num,1)[0]

            motif_assign_vec[node]=picked_motif
            ecount_matrix[labels[node]][picked_motif]+=1
            motif_count[picked_motif]+=1
            if old_motif!=picked_motif:converge=False

        if verbose==True:
            print ecount_matrix
    A=np.zeros(shape=(motif_num,w))
    for i in range(motif_num):
        for j in range(w):
            A[i][j]=ecount_matrix[j][i]/float(motif_count[i])
    return A,motif_assign_vec,ecount_matrix


def filp_noise_node(matrix,motif_assign_vec):
    neigh_motif=None
    n=len(matrix)
    for i in range(n):
        cur_motif=motif_assign_vec[i]
        neigh_motif=None
        filp=True
        for j in range(n):
            if matrix[i][j]==0:continue
            if motif_assign_vec[j]==cur_motif:
                filp=False
                break
            if neigh_motif!=None and motif_assign_vec[j]!=neigh_motif:
                filp=False
                break
            neigh_motif=motif_assign_vec[j]
        if filp==True:
            s=0
            cur_motif=motif_assign_vec[i]
            for j in range(n):
                if matrix[j][i]==0:continue
                s+=1
                if motif_assign_vec[j]==cur_motif:
                    filp=False
                    break
                if neigh_motif!=None and motif_assign_vec[j]!=neigh_motif:
                    filp=False
                    break
                neigh_motif=motif_assign_vec[j]
            if s==0:filp=False
        if filp==True:motif_assign_vec[i]=neigh_motif
def remove_uncontiguous_link(matrix,motif_assign_vec):
    n=len(matrix)
    for i in range(n):
        for j in range(n):
            if matrix[i][j]!=0 and motif_assign_vec[i]!=motif_assign_vec[j]:
                matrix[i][j]=0
def matrix_to_graph(matrix,labels):
    g=Graph(directed=True)
    node_num=len(matrix)
    g.add_vertex(node_num)
    label_property= g.new_vertex_property("string")  
    for i in range(node_num):
        label_property[g.vertex(i)]=labels[i]
    for i in range(node_num):
        for j in range(node_num):
            if matrix[i][j]==1:
                g.add_edge(g.vertex(i),g.vertex(j))
    return g,label_property
def graph_to_matrix(g,label_property):
    node_num=len(list(g.vertices()))
    matrix=[]
    labels=[]
    for i in range(node_num):
        labels.append(label_property[g.vertex(i)])
    for i in range(node_num):
        matrix.append([0]*node_num)
    for e in g.edges():
        matrix[int(e.source())][int(e.target())]=1
    return matrix,labels
def delete_node_set(matrix,labels,motif_assign_vec,delete_indexes):
    new_matrix=np.delete(matrix,list(delete_indexes),0)
    new_matrix=np.delete(new_matrix,list(delete_indexes),1)
    new_labels=np.delete(labels,list(delete_indexes))
    new_motif_assign_vec=np.delete(motif_assign_vec,list(delete_indexes))
    return new_matrix,new_labels,new_motif_assign_vec

def extract_matrix(matrix,labels,component):
    new_id_map={}
    for ind,old_id in enumerate(component):
        new_id_map[old_id]=ind
    component_matrix=[]
    component_labels=[]
    for i in range(len(component)):
        component_matrix.append([0]*len(component))
        component_labels.append(labels[component[i]])
    for i in range(len(component)):
        for j in range(len(component)):
            component_matrix[i][j]=matrix[component[i]][component[j]]
    return component_matrix,component_labels
def extract_graph(matrix,labels,component):
    component_matrix,component_labels=extract_matrix(matrix,labels,component)
    return matrix_to_graph(component_matrix,component_labels)


def matrix_equality(matrix,labels,component1,component2):
    g1,lp1=extract_graph(matrix,labels,component1)
    g2,lp2=extract_graph(matrix,labels,component2)
    return subgraph_isomorphism(g1,g2,vertex_label=(lp1,lp2))
def get_connected_components(matrix,labels):
    g,lp=matrix_to_graph(matrix,labels)
    lc=label_components(g,directed=False)
    cp=lc[0]
    connected_components=[]
    for i in range(len(lc[1])):connected_components.append([])
    for i in range(len(labels)):
        connected_components[cp[i]].append(i)
    return connected_components
def postprocess(matrix,labels,doc_labels,motif_assign_vec):
    frag_workflow=[]
    ##partition graph,remove one step,repeated frag
    remove_uncontiguous_link(matrix,motif_assign_vec)
    ##find connected components
    connected_components=get_connected_components(matrix,labels)
   
    ##delete one step
    delete_indexes=set()
    for c in connected_components:
        if len(c)==1:
            delete_indexes.add(c[0])
    #print delete_indexes
    matrix,labels,motif_assign_vec=delete_node_set(matrix,labels,motif_assign_vec,delete_indexes)
    doc_labels=np.delete(doc_labels,list(delete_indexes))

    ##create frag_workflow
    connected_components=get_connected_components(matrix,labels)
    for i in range(len(connected_components)):frag_workflow.append([])
    for i in range(len(connected_components)):
#        print doc_labels[connected_components[i][0]]
        frag_workflow[i]=[doc_labels[connected_components[i][0]]]
 #   print doc_labels
    #print frag_workflow

    ##
    ###eliminate repeat frag
    remain_frag_indicator=[1]*len(connected_components)
    delete_frag_indexes=[]
    for i in range(len(connected_components)):
        for j in range(i+1,len(connected_components)):
            if len(connected_components[i])!=len(connected_components[j]) or remain_frag_indicator[i]==0:
                continue
            if matrix_equality(matrix,labels,connected_components[i],connected_components[j])!=[]:#equal
                remain_frag_indicator[j]=0
                frag_workflow[i].append(frag_workflow[j][0])
                delete_frag_indexes.append(j)

      

    #delete nodes
    delete_indexes=[]
    for i in range(len(connected_components)):
        if remain_frag_indicator[i]==0:
            for j in range(len(connected_components[i])):
                delete_indexes.append(connected_components[i][j])
    ##
    new_indexes_map=[i for i in range(len(matrix))]
    for i in range(len(matrix)):
        new_indexes_map[i]-=len(filter(lambda(x):x<=i,delete_indexes))
    #print new_indexes_map
#    print connected_components
    for i in range(len(connected_components)):
        for j in range(len(connected_components[i])):
            connected_components[i][j]=new_indexes_map[connected_components[i][j]]
  ##
    connected_components=np.delete(connected_components,list(delete_frag_indexes))
    print frag_workflow
    frag_workflow=np.delete(frag_workflow,list(delete_frag_indexes))
    print frag_workflow
    ##
    ##

    matrix,labels,motif_assign_vec=delete_node_set(matrix,labels,motif_assign_vec,delete_indexes)
    print len(matrix),max(new_indexes_map),len(labels)
    #connected_components=get_connected_components(matrix,labels)
    #print connected_components
    print 'number of frag:',np.count_nonzero(remain_frag_indicator)
    ###
    for i in range(len(frag_workflow)): 
       frag_workflow[i]=list(set(frag_workflow[i]))
  

    return matrix,labels,motif_assign_vec,connected_components,frag_workflow

    #g,lp=matrix_to_graph(matrix,labels)
    #matrix,labels=graph_to_matrix(g,lp)


def get_distance_mat(g,labels):
    distance_matrix= shortest_distance(g,directed=False)
    d_m=[]
    #print len(labels)
    for i in  range(len(labels)):
        d_m.append(distance_matrix[i])
    return d_m
def load_pickle(f_name):
    with open(f_name, 'rb') as handle:
        b = json.load(handle)
    return b[0],b[1],b[2]
def get_index_label_dict(labels):
    index_label_dict={}
    for ind,la in enumerate(labels):
        index_label_dict[la]=ind
    return index_label_dict

def combine_matrix_to_one(matrix_list,labels_list):
    matrix=[]
    labels=[]
    for i in range(len(matrix_list)):
        new_matrix=matrix_list[i]
        matrix_node_num=len(matrix)
        new_matrix_node_num=len(new_matrix)
        for j in range(matrix_node_num):
            t=[0]*new_matrix_node_num
            matrix[j].extend(t)
        
        for j in range(new_matrix_node_num):
            new_matrix[j]=([0]*matrix_node_num)+new_matrix[j]
        matrix.extend(new_matrix)
        labels.extend(labels_list[i])
    return matrix,labels

def group_by_topic(matrix,labels,motif_assign_vec,motif_num):
    connected_components=get_connected_components(matrix,labels)
    matrix_list=[]#each entry is matrix with same topic
    labels_list=[]
    for i in range(motif_num):
        matrix_list.append([])
        labels_list.append([])
    for c in connected_components:
        m,l=extract_matrix(matrix,labels,c)
        matrix_list[motif_assign_vec[c[0]]].append(m)
        labels_list[motif_assign_vec[c[0]]].append(l)
    for i in range(motif_num):
        matrix_list[i],labels_list[i]=combine_matrix_to_one(matrix_list[i],labels_list[i])
    return matrix_list,labels_list

pickle_name=sys.argv[1]
output_dir='./output/'
motif_num=18
print 'load pickle'
matrix,labels,doc_labels=load_pickle(output_dir+pickle_name)
print 'finish load pickle'
g,lp=matrix_to_graph(matrix,labels)
index_label_dict=get_index_label_dict(set(labels))
int_labels=[index_label_dict[la] for la in labels]
d_m=get_distance_mat(g,labels)

A,motif_assign_vec,ecount_matrix=motif_lda(int_labels,d_m,len(set(labels)),len(labels),motif_num,100,verbose=True)

plot_dir(matrix,labels,doc_labels,motif_assign_vec=motif_assign_vec,dirname=output_dir+'/summary/')


filp_noise_node(matrix,motif_assign_vec)
matrix,labels,motif_assign_vec,connected_components,frag_workflow=postprocess(matrix,labels,doc_labels,motif_assign_vec)
plot_dir(matrix,labels,[],motif_assign_vec=motif_assign_vec,components=connected_components,dirname=output_dir+'/fragments/')

