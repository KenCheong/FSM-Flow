from utility import *
import numpy as np
from numpy import argmax,log
from random import randint,uniform,shuffle
import numpy as np
import math 
import scipy as sc


def DE_perplexity(distance_matrix,ecount_matrix,topic_count,node_group,labels,topic_assign_vec,topic_num,alpha,beta):
    label_num=len(set(labels))
    A=np.zeros(shape=(topic_num,label_num))
    for i in range(topic_num):
        for j in range(label_num):
            A[i][j]=(ecount_matrix[j][i]+beta)/(float(topic_count[i])+label_num*beta)
    p=0.0
    for i in range(len(labels)):
        neigh_port=alpha
        s=topic_num*alpha
        for j in node_group[i]:
            s+=sim(distance_matrix[i][j])
            if topic_assign_vec[j]==topic_assign_vec[i]:
                neigh_port+=(sim(distance_matrix[i][j]))
 
        p+=-log(A[topic_assign_vec[i]][labels[i]])-log(float(neigh_port/s))
    return p


def sim(dis):
    var=100
    mean=0
    return 1.0/(dis+1)

def get_topic_portion(node,node_group,distance_matrix,labels,topic_assign_vec,topic_num,ecount_matrix,topic_count,alpha=0.05,beta=0.05):
    label_num=len(set(labels))
    topic_portion=[0]*topic_num
    topic_neigh_portion=[0]*topic_num
    for i in range(topic_num):
        topic_portion[i]=((float(ecount_matrix[labels[node]][i])+alpha)  )/(topic_count[i]+alpha*label_num)
    normalize(topic_portion)
    s=0.0
    for i in node_group[node]:
        s+=sim(distance_matrix[node][i])
        topic_neigh_portion[topic_assign_vec[i]]+=(sim(distance_matrix[node][i]))
    for i in range(topic_num):
        topic_neigh_portion[i]+=beta
    normalize(topic_neigh_portion)
    for i in range(topic_num):
        topic_portion[i]*=(topic_neigh_portion[i])
    normalize(topic_portion)
    return topic_portion

def DE_gibb_sampling(labels,distance_matrix,topic_num,iter_num,distance_threshold=5,verbose=False,alpha=0.05,beta=0.05):
    label_num=len(set(labels))
    node_num=len(labels)
    topic_assign_vec=[0]*node_num
    topic_portion=[0]*topic_num
   
    ecount_matrix=np.zeros(shape=(label_num,topic_num),dtype=np.int)
    topic_count=np.zeros(shape=(topic_num))
    node_group=[]
     
   
    #initialize
    for ii in range(node_num):
        picked_topic=np.random.choice(topic_num,1)[0]
        topic_assign_vec[ii]=picked_topic
        ecount_matrix[labels[ii]][picked_topic]+=1
        topic_count[picked_topic]+=1
    ######
    ######inintialize node group
    for ii in range(node_num):
        node_group.append([])
        for j in range(node_num):
            if ii!=j and distance_matrix[ii][j]<distance_threshold and topic_assign_vec[j]!=-1:node_group[ii].append(j)
    ###
    for i in range(iter_num): 
        print 'iter num:'+str(i)
        scan_order=range(node_num)
        shuffle(scan_order)
               
        for node in scan_order:
            if topic_assign_vec[node]==-1:continue
            old_topic=topic_assign_vec[node]
            ecount_matrix[labels[node]][old_topic]-=1
            topic_count[old_topic]-=1

            topic_portion=get_topic_portion(node,node_group,distance_matrix,labels,topic_assign_vec,topic_num,ecount_matrix,topic_count,alpha=alpha,beta=beta)
            picked_topic=np.random.choice(topic_num,1,p=topic_portion)[0]

            topic_assign_vec[node]=picked_topic
            ecount_matrix[labels[node]][picked_topic]+=1
            topic_count[picked_topic]+=1
        if verbose==True:
#            print ecount_matrix
            print 'perplexity:',DE_perplexity(distance_matrix,ecount_matrix,topic_count,node_group,labels,topic_assign_vec,topic_num,alpha,beta)


    return topic_assign_vec

