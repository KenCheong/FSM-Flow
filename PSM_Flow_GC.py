from utility import *
from sklearn.manifold import MDS
import numpy as np
from scipy.stats import multivariate_normal
from numpy import argmax,log
from random import randint,uniform,shuffle
import math 
import scipy as sc
def GC_perplexity(components,point_vec,ecount_matrix,topic_count,labels,frag_assign_vec,topic_assign_vec,frag_mean,frag_var,topic_num,beta):
    label_num=len(set(labels))
    A=np.zeros(shape=(topic_num,label_num))
    for i in range(topic_num):
        for j in range(label_num):
            A[i][j]=(ecount_matrix[j][i]+beta)/(float(topic_count[i])+label_num*beta)
    p=0.0
    for i in range(len(components)):
        for j in range(len(components[i])):
           p+=-log(A[topic_assign_vec[components[i][j]]][labels[components[i][j]]])-log(multivariate_normal.pdf(point_vec[i][j], mean=frag_mean[i][frag_assign_vec[i][j]], cov=frag_var[i][frag_assign_vec[i][j]],allow_singular=True))
           

    return p



def get_frag_portion(node,component,frag_group,frag_topic,point_vec,frag_mean,frag_var,labels,frag_num,topic_assign_vec,topic_num,ecount_matrix,topic_count,beta=0.05):

    label_num=len(set(labels))
    frag_portion=[0]*frag_num
    for i in range(frag_num):
        frag_portion[i]=multivariate_normal.pdf(point_vec[node], mean=frag_mean[i], cov=frag_var[i],allow_singular=True)*((float(ecount_matrix[labels[component[node]]][frag_topic[i]])+beta)/(topic_count[frag_topic[i]]+beta*label_num))
    normalize(frag_portion)
    return frag_portion
def get_gaussian_topic_portion(frag,labels,topic_assign_vec,topic_num,ecount_matrix,topic_count,beta=0.05):
    label_num=len(set(labels))
    topic_portion=[0]*topic_num
    if len(frag)==0:
        for i in range(topic_num):
            topic_portion[i]=1.0/topic_num
        return topic_portion
    s=1.0
    for i  in range(topic_num):
        s=1.0
        for node in frag:
            s*=(float(ecount_matrix[labels[node]][i])+beta)/(topic_count[i]+beta*label_num)
        topic_portion[i]=s
    normalize(topic_portion)
    return topic_portion
def mean(frag,point_vec):
    s=[0.0]*len(point_vec[0])
    n=float(len(frag))
    if n==0:return s
    for d in range(len(point_vec[0])):
        for f_id in frag:
            s[d]+=point_vec[f_id][d]
    for d in range(len(point_vec[0])):
        s[d]/=(n)
    return s

def var(frag,point_vec,dimensionlen):
    a=1.0
    t=[]
    for i in range(dimensionlen):
        t.append([0.0]*dimensionlen)
        t[i][i]=1.0/((1.0/a)+len(frag))
    return t
    
def GC_LDA(labels,point_vec,components,topic_num,iter_num,component_cluster_num=[],beta=0.05):
    node_num= len(labels)
    label_num= len(set(labels))

    component_num=len(components)
    topic_assign_vec=[0]*node_num
    topic_portion=[0]*topic_num
    frag_assign_vec=[]
    frag_portion=[]
    frag_topic=[]
    frag_group=[]
    new_frag_group=[]
    dimensionlen=len(point_vec[0][0])
    frag_mean=[]
    frag_var=[]
    frag_group_nums=[]
    for i in range(component_num):
        frag_assign_vec.append([])
        frag_topic.append([])
        frag_mean.append([])
        frag_var.append([])
        frag_group.append([])
        new_frag_group.append([])
        if component_cluster_num!=[]:
            frag_group_nums.append(component_cluster_num[i])
        else:
            frag_group_nums.append(len(components[i]))
    for i in range(component_num):
        frag_assign_vec[i]=[0]*len(components[i])
        frag_topic[i]=[0]*len(components[i])
        for j in range(frag_group_nums[i]):
            frag_mean[i].append([0.0]*len(point_vec[0]))
            frag_group[i].append([])
            new_frag_group[i].append([])

    t=[]
    for i in range(dimensionlen):
        t.append([0.0]*dimensionlen)
        t[i][i]=1.0
    for i in range(component_num):
        for j in range(len(frag_mean[i])):
            frag_var[i].append(t)
   
    ecount_matrix=np.zeros(shape=(label_num,topic_num),dtype=np.int)
    topic_count=np.zeros(shape=(topic_num))

    #initialize
    ###ini frag assign
    for i in range(component_num):
        for j in range(len(components[i])):
            picked_frag=np.random.choice(frag_group_nums[i],1)[0]
            frag_group[i][picked_frag].append(j)
    #init topic
    for i in range(component_num):
        for k in range(frag_group_nums[i]):
            picked_topic=np.random.choice(topic_num,1)[0]
            frag_topic[i][k]=picked_topic
            for j in frag_group[i][k]:
                ecount_matrix[labels[components[i][j]]][picked_topic]+=1
                topic_assign_vec[components[i][j]]=picked_topic
            topic_count[picked_topic]+=len(frag_group[i][k])
            frag_mean[i][k]=mean(frag_group[i][k],point_vec[i])
    s=0
    ######
    for i in range(iter_num): 
        scan_order=range(node_num)
        shuffle(scan_order)
        new_frag_group=[]
        for c in range(component_num):
            new_frag_group.append([])
            for f in range(len(components[c])):new_frag_group[c].append([])
               
   
      ##update mean var 
        for c in range(component_num):
            for f in range(frag_group_nums[c]):
                frag_mean[c][f]=np.random.multivariate_normal(mean(frag_group[c][f],point_vec[c]),var(frag_group[c][f],point_vec[c],dimensionlen) , 1)[0]

        ###
        ##update doc membership
        for c in range(component_num):
            for group_node in range(len(components[c])):
                frag_portion=get_frag_portion(group_node,components[c],frag_group[c],frag_topic[c],point_vec[c],frag_mean[c],frag_var[c],labels,frag_group_nums[c],topic_assign_vec,topic_num,ecount_matrix,topic_count,beta=beta)
                picked_frag=np.random.choice(frag_group_nums[c],1,p=frag_portion)[0]
                new_frag_group[c][picked_frag].append(group_node)
                frag_assign_vec[c][group_node]=picked_frag
        frag_group=new_frag_group

        for c in range(component_num):
            for f in range(frag_group_nums[c]):
                topic_portion=get_gaussian_topic_portion([components[c][b] for b in frag_group[c][f]],labels,topic_assign_vec,topic_num,ecount_matrix,topic_count,beta=beta)
                picked_topic=np.random.choice(topic_num,1,p=topic_portion)[0]
                frag_topic[c][f]=picked_topic
                for group_node in frag_group[c][f]:
                    old_topic=topic_assign_vec[components[c][group_node]]
                    ecount_matrix[labels[components[c][group_node]]][old_topic]-=1
                    topic_count[old_topic]-=1
                    topic_assign_vec[components[c][group_node]]=picked_topic
                    ecount_matrix[labels[components[c][group_node]]][picked_topic]+=1
                    topic_count[picked_topic]+=1
           
       


        print "iter_num",i
        print 'perplexity:',GC_perplexity(components,point_vec,ecount_matrix,topic_count,labels,frag_assign_vec,topic_assign_vec,frag_mean,frag_var,topic_num,beta)

    return topic_assign_vec


