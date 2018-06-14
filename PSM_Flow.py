from utility import *
from scipy.stats import poisson

from sklearn.manifold import MDS
import numpy as np
from scipy.stats import multivariate_normal
from numpy import argmax,log
from random import randint,uniform,shuffle
import math 
import scipy as sc

def update_frag_topic(ecount_matrix,components,frag_group,frag_topic,topic_num,topic_count,topic_assign_vec,labels):
    for c in range(len(frag_topic)):
        for i in range(len(frag_topic[c])):
            topic_portion=get_gaussian_topic_portion([components[c][b] for b in frag_group[c][i]],labels,topic_assign_vec,topic_num,ecount_matrix,topic_count)[:-1]
            '''
            topic_portion=[0.0]*topic_num
            total_nodes=float(sum(topic_count))
            for j in range(topic_num):
                topic_portion[j]=topic_count[j]/total_nodes
            '''
            normalize(topic_portion)
            picked_topic=np.random.choice(topic_num,1,p=topic_portion)[0]
            frag_topic[c][i]=picked_topic
    return frag_topic
def update_frag_group(frag_group,frag_topic,frag_mean,frag_var,frag_group_nums,dimensionlen):
    t=[]
    for i in range(dimensionlen):
        t.append([0.0]*dimensionlen)
        t[i][i]=1.0

    for c in range(len(frag_group)):
        remove_list=[]
        for f in range(len(frag_group[c])):
            if len(frag_group[c][f])==0:
                remove_list.append(f)
#        print 'f',np.delete(frag_group[c] ,remove_list,axis=0).tolist(),frag_group[c]
        frag_group[c]=np.delete(frag_group[c] ,remove_list,axis=0).tolist()
        frag_group[c].append([])
        #print 'sd',frag_group[c]
        frag_topic[c]=np.delete(frag_topic[c] ,remove_list,axis=0).tolist()
        frag_topic[c].append(0)
        frag_mean[c]=np.delete(frag_mean[c] ,remove_list,axis=0).tolist()
        frag_mean[c].append([0.0]*dimensionlen)
        frag_var[c]=np.delete(frag_var[c] ,remove_list,axis=0).tolist()
        frag_var[c].append(t)
        frag_group_nums[c]=len(frag_group[c])

def add_topic(ecount_matrix,topic_count,topic_assign_vec,topic_size_list,topic_num):
    ec=list(ecount_matrix)
    ec.append([0]*len(ecount_matrix[0]))
    tc=list(topic_count)
    tc.append(0)
    ts=list(topic_size_list)
    ts.append(1)
    return ec,tc,ts,topic_num+1
def remove_empty_topic(ecount_matrix,topic_count,topic_assign_vec,topic_size_list,topic_num):
    remove_list=[]
    for i in range(topic_num):
        if topic_count[i]==0:
            remove_list.append(i)

    for j in range(len(topic_assign_vec)):
        t=0
        for i in remove_list:
            if topic_assign_vec[j]>i:
                t+=1
        topic_assign_vec[j]-=t

 


    return np.delete( ecount_matrix,remove_list,axis=0).tolist(),np.delete(topic_count ,remove_list,axis=0).tolist(),np.delete(topic_size_list ,remove_list,axis=0).tolist(),(topic_num-len(remove_list))
    
def update_topic_size(topic_size_list,frag_group,frag_topic):
    topic_num=len(topic_size_list)
    topic_count_list=[0]*topic_num
    for t in range(topic_num):
        topic_size_list[t]=0.0
    for c in range(len(frag_group)):
        for f in range(len(frag_group[c])):
            if len(frag_group[c][f])<2:continue
            topic_count_list[frag_topic[c][f]]+=1.0
            topic_size_list[frag_topic[c][f]]+=len(frag_group[c][f])
    for t in range(topic_num):
        if topic_count_list[t]<2:continue
        topic_size_list[t]/=topic_count_list[t]


#def GC_perplexity(components,point_vec,ecount_matrix,topic_count,labels,frag_assign_vec,topic_assign_vec,frag_mean,frag_var,topic_num,beta):
def get_dis_from_point(dis_matrix,latent_point_vec):
    D=np.zeros(shape=(len(dis_matrix),len(dis_matrix)))
    for i in range(len(dis_matrix)):
        for j in range(i+1,len(dis_matrix)):
            D[i][j]=np.linalg.norm(latent_point_vec[i]-latent_point_vec[j])
            D[j][i]=D[i][j]
    return D
def GC_perplexity(components,latent_point_vec,ecount_matrix,topic_count,labels,frag_assign_vec,topic_assign_vec,frag_mean,frag_var,topic_num,dis_matrices,beta):
    label_num=len(set(labels))
    A=np.zeros(shape=(topic_num,label_num))
    for i in range(topic_num):
        for j in range(label_num):
            A[i][j]=(ecount_matrix[i][j]+beta)/(float(topic_count[i])+label_num*beta)
    p=0.0
    for i in range(len(components)):
        D=get_dis_from_point(dis_matrices[i],latent_point_vec[i])
        for j in range(len(components[i])):
           p+=-log(A[topic_assign_vec[components[i][j]]][labels[components[i][j]]])-log(multivariate_normal.pdf(latent_point_vec[i][j], mean=frag_mean[i][frag_assign_vec[i][j]], cov=frag_var[i][frag_assign_vec[i][j]],allow_singular=True))
        '''
        for j in range(len(components[i])):
            for k in range(len(components[i])):
               p+=-log(single_normal_pdf(D[j][k],dis_matrices[i][j][k],1))
        '''

    return np.exp(p/len(labels))
def get_geodesic_vec(latent_point_vec,dis_matrix,group_node):
    gradient=0.0
#    print latent_point_vec[0]-latent_point_vec[1]
    for i in range(len(dis_matrix)):
        if i==group_node:continue
        e_dis=np.linalg.norm(latent_point_vec[group_node]-latent_point_vec[i])
    #    if e_dis<=0.01:continue
        gradient+=((e_dis-dis_matrix[i][group_node])/e_dis)*(latent_point_vec[group_node]-latent_point_vec[i])

    return latent_point_vec[group_node]-2*(gradient)/len(dis_matrix)
def sample_vec(group_node,latent_point_vec,mean,topic_var,dis_matrix,sample_num=7):
    dimensionlen=len(mean)

    #vec_samples=np.random.multivariate_normal(mean,var([],[],dimensionlen,coef=1) , sample_num)
    vec_samples=np.random.multivariate_normal(mean,topic_var , sample_num)
    portion=[1.0]*sample_num
    for i in range(sample_num):
        for n in range(len(dis_matrix)):
            if group_node==n or dis_matrix[group_node][n]>7:continue
    #        if single_normal_pdf(np.linalg.norm(vec_samples[i]-latent_point_vec[n]),dis_matrix[group_node][n],1)==0:
     #           print vec_samples[i],latent_point_vec[n],dis_matrix[group_node][n]
            portion[i]*=single_normal_pdf(np.linalg.norm(vec_samples[i]-latent_point_vec[n]),dis_matrix[group_node][n],1)
    #    portion[i]=np.exp(portion[i])
    normalize(portion)
    #print portion
    pick_vec=np.random.choice(sample_num,1,p=portion)[0]
    return vec_samples[pick_vec]
def single_normal_pdf(x,m,std):
    return np.exp(-((x-m)**2)/2.0)

def update_position_vec(latent_point_vec,dis_matrices,frag_group,frag_mean):
    dimensionlen=len(latent_point_vec[0][0])
    tv=np.eye(dimensionlen)
    for c in range(len(frag_group)):
        node_num=float(len(latent_point_vec[c]))
        for f in range(len(frag_group[c])):
            for group_node in frag_group[c][f]:
                #geodesic_vec=get_geodesic_vec(latent_point_vec[c],dis_matrices[c],group_node)
                #latent_point_vec[c][group_node]=np.random.multivariate_normal((np.array(frag_mean[c][f])+node_num*geodesic_vec)/(node_num+1),var(frag_group[c][f],latent_point_vec[c],dimensionlen,coef=node_num+1) , 1)[0]
                v=sample_vec(group_node,latent_point_vec[c],frag_mean[c][f],tv,dis_matrices[c])
                latent_point_vec[c][group_node]=v


                #latent_point_vec[c][group_node]=np.random.multivariate_normal((np.array(frag_mean[c][f])+geodesic_vec)/2.0,var([],[],dimensionlen,coef=0.5) , 1)[0]



def get_frag_portion(node,matrix,topic_size_list,component,frag_group,frag_topic,point_vec,frag_mean,frag_var,labels,frag_num,topic_assign_vec,topic_num,ecount_matrix,topic_count,topic_masks,beta=0.05):

    label_num=len(set(labels))
    frag_portion=[0]*frag_num
    f_len=[]
    old_frag=0
    for i in range(frag_num):
        for j in frag_group[i]:
            if j==node:old_frag=i
        f_len.append(len(frag_group[i]))

    for i in range(frag_num):
        #frag_portion[i]=multivariate_normal.pdf(point_vec[node], mean=frag_mean[i], cov=frag_var[i],allow_singular=True)*((float(ecount_matrix[labels[component[node]]][frag_topic[i]])+beta)/(topic_count[frag_topic[i]]+beta*label_num))
        connected=False
        if len(frag_group[i])>0:
            for j in frag_group[i]:
                if matrix[component[node]][component[j]]==1 or matrix[component[j]][component[node]]==1:
                    connected=True
                    break
        if connected==False and len(frag_group[i])>0:
            frag_portion[i]=0.0
            continue
#        print (topic_size_list[frag_topic[i]]**len(frag_group[i]))/float(np.math.factorial(len(frag_group[i])))*np.exp(-topic_size_list[frag_topic[i]])

        if len(frag_group[i])==0:
            frag_portion[i]=normal_pdf(frag_mean[i],point_vec[node])*((float(ecount_matrix[frag_topic[i]][labels[component[node]]])+beta)/(topic_count[frag_topic[i]]+beta*label_num))*(topic_size_list[frag_topic[i]]**1)/float(np.math.factorial(1))*np.exp(-topic_size_list[frag_topic[i]])
        else:
            if i!=old_frag:
                frag_portion[i]=normal_pdf(frag_mean[i],point_vec[node])*((float(ecount_matrix[frag_topic[i]][labels[component[node]]])+beta)/(topic_count[frag_topic[i]]+beta*label_num))
#*(topic_size_list[frag_topic[i]]**(f_len[i]+1))/float(np.math.factorial(f_len[i]+1))*np.exp(-topic_size_list[frag_topic[i]])
            else:
                frag_portion[i]=normal_pdf(frag_mean[i],point_vec[node])*((float(ecount_matrix[frag_topic[i]][labels[component[node]]])+beta)/(topic_count[frag_topic[i]]+beta*label_num))
#*(topic_size_list[frag_topic[i]]**(f_len[i]))/float(np.math.factorial(f_len[i]))*np.exp(-topic_size_list[frag_topic[i]])
    #if sum(frag_portion)==0:
        #print frag_portion
#        print "zeros:",normal_pdf(frag_mean[i],point_vec[node]),((float(ecount_matrix[frag_topic[i]][labels[component[node]]])+beta)/(topic_count[frag_topic[i]]+beta*label_num)),(topic_size_list[frag_topic[i]]**f_len[i])/float(np.math.factorial(f_len[i]))*np.exp(-topic_size_list[frag_topic[i]]),normal_pdf(frag_mean[i],point_vec[node])*((float(ecount_matrix[frag_topic[i]][labels[component[node]]])+beta)/(topic_count[frag_topic[i]]+beta*label_num))*(topic_size_list[frag_topic[i]]**f_len[i])/float(np.math.factorial(len(frag_group[i])))*np.exp(-topic_size_list[frag_topic[i]])

    if sum(frag_portion)==0 and len(frag_portion)<3:
        for i in range(len(frag_portion)):
            frag_portion[0]=1



    normalize(frag_portion)
    return frag_portion


def get_gaussian_topic_portion(frag,labels,topic_assign_vec,topic_num,ecount_matrix,topic_count,beta=0.5,tau=50):
    label_num=len(set(labels))
    topic_portion=[0]*(topic_num+1)
    total_nodes=len(topic_assign_vec)
    tau=float(tau)
    if len(frag)==0:
        for i in range(topic_num):
            topic_portion[i]=1.0/topic_num
        topic_portion[-1]=tau/(total_nodes+tau)
        normalize(topic_portion)
        return topic_portion
    s=1.0
    for i  in range(topic_num):
        s=1.0
        for node in frag:
#            print ecount_matrix[labels[node]],i
            s*=(float(ecount_matrix[i][labels[node]])+beta)/(topic_count[i]+beta*label_num)
        s*=((topic_count[i]+tau)/(total_nodes+tau))
        topic_portion[i]=s
    normalize(topic_portion)
    topic_portion[-1]=(tau/(total_nodes+tau))*((beta/(label_num*beta))**len(frag))
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

def var(frag,point_vec,dimensionlen,coef=1.0):
    a=coef
    t=[]
    for i in range(dimensionlen):
        t.append([0.0]*dimensionlen)
        t[i][i]=1.0/((1.0/a)+len(frag))
    return t
def update_transition(matrix,labels,components,frag_group,frag_topic,transition_count_matrices,pair_count_matrices,ind_label=[]):    
    component_num=len(components)
    for t in range(len(transition_count_matrices)):
        for i in range(len(transition_count_matrices[t])):
            for j in range(len(transition_count_matrices[t][i])):
                transition_count_matrices[t][i][j]=0
                pair_count_matrices[t][i][j]=0
    for i in range(component_num):
        for k in range(len(frag_group[i])):
            for j in frag_group[i][k]:
                for d in frag_group[i][k]:
                    if j==d :continue
                    if matrix[components[i][j]][components[i][d]]==1:    
                        transition_count_matrices[frag_topic[i][k]][labels[components[i][j]]][labels[components[i][d]]]+=1
                        t=0
                        if float(pair_count_matrices[frag_topic[i][k]][labels[components[i][j]]][labels[components[i][d]]])!=0:
                            t=float(pair_count_matrices[frag_topic[i][k]][labels[components[i][j]]][labels[components[i][d]]])/float(pair_count_matrices[frag_topic[i][k]][labels[components[i][j]]][labels[components[i][d]]])
                        print  ind_label[labels[components[i][j]]],ind_label[labels[components[i][d]]],frag_topic[i][k],t
                    pair_count_matrices[frag_topic[i][k]][labels[components[i][j]]][labels[components[i][d]]]+=1
 
    '''
    for t in range(len(transition_count_matrices)):
        for i in range(len(transition_count_matrices[0])):
            for j in range(len(transition_count_matrices[0])):
                if transition_count_matrices[t][i][j]>0:
                    print  ind_label[i],ind_label[j]
    '''
 
def PSM_Flow(matrix,labels,dis_matrices,components,topic_num,iter_num,component_cluster_num=[],beta=0.05,dimensionlen=10,ind_label=[],tau=0.5):
    node_num= len(labels)
    print set(labels)
    label_num= len(set(labels))
    dis_point_vec=[]
    latent_point_vec=[]
    component_num=len(components)
    topic_assign_vec=[0]*node_num
    topic_portion=[0]*topic_num
    topic_size_list=[5]*topic_num

    perplexity_list=[]
    frag_assign_vec=[]
    frag_portion=[]
    frag_topic=[]
    frag_group=[]
    new_frag_group=[]
#    dimensionlen=len(point_vec[0][0])
    frag_mean=[]
    frag_var=[]
    frag_group_nums=[]
    for i in range(component_num):
        frag_assign_vec.append([])
        frag_topic.append([])
        frag_mean.append([])
        latent_point_vec.append([])
        frag_var.append([])
        frag_group.append([])
        new_frag_group.append([])
        if component_cluster_num!=[]:
            frag_group_nums.append(component_cluster_num[i])
        else:
            frag_group_nums.append(len(components[i]))
        for j in range(len(components[i])):
            #latent_point_vec[i].append([0.0]*dimensionlen)
#            latent_point_vec[i].append(np.zeros(dimensionlen))
            latent_point_vec[i].append(np.random.multivariate_normal(np.zeros(dimensionlen),var([],[],dimensionlen) , 1)[0])

    for i in range(component_num):
        frag_assign_vec[i]=[0]*len(components[i])
        frag_topic[i]=[0]*len(components[i])
        for j in range(frag_group_nums[i]):
            #frag_mean[i].append([0.0]*len(point_vec[0]))
            frag_mean[i].append([0.0]*dimensionlen)
            frag_group[i].append([])
            new_frag_group[i].append([])

    t=[]
    for i in range(dimensionlen):
        t.append([0.0]*dimensionlen)
        t[i][i]=1.0
    for i in range(component_num):
        for j in range(len(frag_mean[i])):
            frag_var[i].append(t)
   
    ecount_matrix=np.zeros(shape=(topic_num,label_num),dtype=np.int)
    logit_coef_matrix=np.zeros(shape=(label_num,dimensionlen),dtype=np.float)

    topic_masks=np.ones(shape=(topic_num,label_num),dtype=np.int)
    topic_count=np.zeros(shape=(topic_num))

    #initialize
    ###ini frag assign
    for i in range(component_num):
        for j in range(len(components[i])):
            picked_frag=np.random.choice(frag_group_nums[i],1)[0]
            frag_group[i][picked_frag].append(j)
    update_position_vec(latent_point_vec,dis_matrices,frag_group,frag_mean)
    #print latent_point_vec[0]
    #init topic
    for i in range(component_num):
        for k in range(frag_group_nums[i]):
            picked_topic=np.random.choice(topic_num,1)[0]
            frag_topic[i][k]=picked_topic
            for j in frag_group[i][k]:
                ecount_matrix[picked_topic][labels[components[i][j]]]+=1
                topic_assign_vec[components[i][j]]=picked_topic
            topic_count[picked_topic]+=len(frag_group[i][k])
            frag_mean[i][k]=mean(frag_group[i][k],latent_point_vec[i])
    s=0
    ######
    for i in range(iter_num): 
        scan_order=range(node_num)
        shuffle(scan_order)
        new_frag_group=[]
        for c in range(component_num):
            new_frag_group.append([])
            for f in range(frag_group_nums[c]):new_frag_group[c].append([])
               
   
        update_position_vec(latent_point_vec,dis_matrices,frag_group,frag_mean)
      ##update mean var 
        for c in range(component_num):
            for f in range(frag_group_nums[c]):
                frag_mean[c][f]=np.random.multivariate_normal(mean(frag_group[c][f],latent_point_vec[c]),var(frag_group[c][f],latent_point_vec[c],dimensionlen,coef=1.0/(max(len(frag_group[c][f]),1))) , 1)[0]

        ###
        ##update doc membership
        for c in range(component_num):
            for group_node in range(len(components[c])):
                #frag_portion=get_frag_portion(group_node,components[c],frag_group[c],frag_topic[c],latent_point_vec[c],frag_mean[c],frag_var[c],labels,frag_group_nums[c],topic_assign_vec,topic_num,ecount_matrix,topic_count,topic_masks,beta=beta)
                frag_portion=get_frag_portion(group_node,matrix,topic_size_list,components[c],frag_group[c],frag_topic[c],latent_point_vec[c],frag_mean[c],[],labels,frag_group_nums[c],topic_assign_vec,topic_num,ecount_matrix,topic_count,topic_masks,beta=beta)

                picked_frag=np.random.choice(frag_group_nums[c],1,p=frag_portion)[0]
                new_frag_group[c][picked_frag].append(group_node)
                frag_assign_vec[c][group_node]=picked_frag
        frag_group=new_frag_group

        for c in range(component_num):
            for f in range(frag_group_nums[c]):
                topic_portion=get_gaussian_topic_portion([components[c][b] for b in frag_group[c][f]],labels,topic_assign_vec,topic_num,ecount_matrix,topic_count,beta=beta,tau=tau)
#                print topic_portion
                picked_topic=np.random.choice(topic_num+1,1,p=topic_portion)[0]
                if picked_topic==topic_num:#open new topic
                    ecount_matrix,topic_count,topic_size_list,topic_num=add_topic(ecount_matrix,topic_count,topic_assign_vec,topic_size_list,topic_num)


                frag_topic[c][f]=picked_topic
                for group_node in frag_group[c][f]:
                    old_topic=topic_assign_vec[components[c][group_node]]
                    ecount_matrix[old_topic][labels[components[c][group_node]]]-=1
                    topic_count[old_topic]-=1
                    topic_assign_vec[components[c][group_node]]=picked_topic
                    ecount_matrix[picked_topic][labels[components[c][group_node]]]+=1
                    topic_count[picked_topic]+=1
           
       
        #update_topic_masks(ecount_matrix,topic_masks,jump_prop=0.01)

        ecount_matrix,topic_count,topic_size_list,topic_num=remove_empty_topic(ecount_matrix,topic_count,topic_assign_vec,topic_size_list,topic_num)
        frag_topic= update_frag_topic(ecount_matrix,components,frag_group,frag_topic,topic_num,topic_count,topic_assign_vec,labels)
#        update_frag_group(frag_group,frag_topic,frag_mean,frag_var,frag_group_nums,dimensionlen)
     #   print 'f',frag_group[1],frag_group[0]

        update_topic_size(topic_size_list,frag_group,frag_topic)
    #    print topic_size_list
        print "iter_num",i
     #   print np.array(ecount_matrix).tolist()
#        print frag_group[2]
        print "topic_num",topic_num
#        print 'effective topic_num:',effective_topic_num(ecount_matrix,15)

        #print topic_masks
        #print 'effective topic number:',effective_topic_num(np.array(ecount_matrix).transpose())
        perplexity_list.append(GC_perplexity(components,latent_point_vec,ecount_matrix,topic_count,labels,frag_assign_vec,topic_assign_vec,frag_mean,frag_var,topic_num,dis_matrices,beta))
        print 'perplexity:',perplexity_list[-1]
    transition_count_matrices=[]
    pair_count_matrices=[]
    for t in range(topic_num):
        transition_count_matrices.append(np.zeros(shape=(label_num,label_num)))
        pair_count_matrices.append(np.zeros(shape=(label_num,label_num)))

    transition_matrices=[]
    '''
    update_transition(matrix,labels,components,frag_group,frag_topic,transition_count_matrices,pair_count_matrices,ind_label=ind_label)
    for t in range(topic_num):
        transition_matrices.append(np.zeros(shape=(label_num,label_num)))
    for t in range(topic_num):
        for i in range(label_num):
            for j in range(label_num):
                if pair_count_matrices[t][i][j]>0:
                    transition_matrices[t][i][j]=float(transition_count_matrices[t][i][j])/float(pair_count_matrices[t][i][j])
    '''
#    print transition_count_matrices[0],pair_count_matrices[0],transition_matrices[0]
 #   print transition_count_matrices[1],pair_count_matrices[1],transition_matrices[1]
#    print transition_matrices[0],transition_matrices[1],transition_matrices[2]
    '''
    for i in range(0,len(matrix)-5):
        for j in range(0,len(matrix)-5):
            if matrix[i][j]==1:
                print  ind_label[labels[i]],ind_label[labels[j]]
    '''
    '''
    for t in range(len(transition_count_matrices)):
        for i in range(len(transition_count_matrices[0])):
            for j in range(len(transition_count_matrices[0])):
                if transition_count_matrices[t][i][j]>0.5:
                    print  ind_label[i],ind_label[j]
    '''


    return topic_assign_vec,latent_point_vec,frag_mean,frag_group,frag_topic,transition_matrices,perplexity_list


