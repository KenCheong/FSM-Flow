from copy import copy, deepcopy
import matplotlib.pyplot as plt

from sklearn.manifold import MDS
from scipy.stats import multivariate_normal
import copy
from graph_tool.all import *
import graphviz as gv
from random import randint,uniform,shuffle
import numpy as np
import math 
from sklearn import decomposition
def filter_nonclose_frag(frag_matrix,frag_labels,cor_matrix,cor_labels):
    frag_matrices,frag_labels_list=get_workflow_list(frag_matrix,frag_labels)
    cor_matrices,cor_labels_list=get_workflow_list(cor_matrix,cor_labels)
    frag_graphs=[]
    cor_graphs=[]


    for i in range(len(frag_matrices)):
        g,lp=matrix_to_graph(frag_matrices[i],frag_labels_list[i])
        frag_graphs.append([g,lp])
    for i in range(len(cor_matrices)):
        g,lp=matrix_to_graph(cor_matrices[i],cor_labels_list[i])
        cor_graphs.append([g,lp])
    frag_freq_array=[0]*len(frag_matrices)
    ##freq
    for i in range(len(frag_matrices)):
        print i
        for j in range(len(cor_matrices)):

            if subgraph_isomorphism(frag_graphs[i][0],cor_graphs[j][0],vertex_label=(frag_graphs[i][1],cor_graphs[j][1]),max_n=3)!=[]:
                frag_freq_array[i]+=1
                continue
    sup_list=[]
    for i in range(len(frag_matrices)):
        print i
        sup_list.append([])
        for j in range(len(frag_matrices)):
            if i==j or len(frag_matrices[i])>len(frag_matrices[j]):continue
            if subgraph_isomorphism(frag_graphs[i][0],frag_graphs[j][0],vertex_label=(frag_graphs[i][1],frag_graphs[j][1]),max_n=3)!=[]:
                sup_list[-1].append(j)
    new_matrices,new_labels=[],[]
    for i in range(len(frag_matrices)):
        close=True
        if frag_freq_array[i]<2:continue
        for j in sup_list[i]:
        #    close=False
         #   break
            if frag_freq_array[i]==frag_freq_array[j]:
                print 'not close',i
                close=False
                break
        if close==True:
            new_matrices.append(frag_matrices[i])
            new_labels.append(frag_labels_list[i])
    print 'close fliter len',len(new_matrices)
    matrix,labels=matrices_to_one_matrix(new_matrices,new_labels)
    return matrix,labels
def delete_node_set(matrix,labels,motif_assign_vec,delete_indexes):
    new_matrix=np.delete(matrix,list(delete_indexes),0)
    new_matrix=np.delete(new_matrix,list(delete_indexes),1)
    new_labels=np.delete(labels,list(delete_indexes))
    new_motif_assign_vec=np.delete(motif_assign_vec,list(delete_indexes))
    return new_matrix,new_labels,new_motif_assign_vec

def plot_subgraph_pattern(transition_matrices,int_labels,ind_label,topic_assign_vec):
    topic_num=len(transition_matrices)
    label_num=len(transition_matrices[0])
    subgraph_matrices=[]
    tran_matrices=[]
    subgraph_labels=[]
    A=np.zeros(shape=(topic_num,label_num))
    for i in range(len(topic_assign_vec)):
        A[topic_assign_vec[i]][int_labels[i]]+=1
    ind_array=[]
    tA=[]
    for i in range(topic_num):
        ind_array.append(i)
    for t in range(topic_num):
        tA=[]
        for i in range(label_num):
            tA.append((i,A[t][i]))
        s=sum(A[t])
        a = sorted(tA, key=lambda x:x[1] , reverse=True)
        print a
        j=0
        cs=0.0
        peak_inds=[]
        while cs/s<0.95:
            print a[j]
            peak_inds.append(a[j][0])
            cs+=a[j][1]
            j+=1
        print "topic",t
        t_labels=[]
        tr_matrix=list(np.zeros(shape=(len(peak_inds),len(peak_inds))))
        tmatrix=list(np.zeros(shape=(len(peak_inds),len(peak_inds))))
        for i in peak_inds:
            t_labels.append(ind_label[i])
        for i in peak_inds:
#            print ind_label
#            print i
            print ind_label[i]
            for j in peak_inds:
                if i==j or  transition_matrices[t][i][j]<0.2:continue
                tmatrix[t_labels.index(ind_label[i])][t_labels.index(ind_label[j])]=1
                tr_matrix[t_labels.index(ind_label[i])][t_labels.index(ind_label[j])]=transition_matrices[t][i][j]

        
        subgraph_labels.extend(t_labels)
        subgraph_matrices.append(tmatrix)
        tran_matrices.append(tr_matrix)
#    print subgraph_matrices
 #   print len(subgraph_matrices[0])
    matrix,_=matrices_to_one_matrix(subgraph_matrices)
    tr_marix,_=matrices_to_one_matrix(tran_matrices)
    print tr_marix

#    print subgraph_matrices
    print matrix
    print ind_label
#    print [ind_label[i] for i in int_labels]
 #   print subgraph_labels
    ##delete one step
    connected_components=get_connected_components(matrix,subgraph_labels)
    delete_indexes=set()
    for c in connected_components:
        if len(c)==1:
            delete_indexes.add(c[0])
    #print delete_indexes
    matrix,subgraph_labels,motif_assign_vec=delete_node_set(matrix,subgraph_labels,topic_assign_vec,delete_indexes)

    plot_graph(matrix,subgraph_labels,[],fname='subgraphs_pattern',transition_matrix=tr_marix)




 


def plot_vec_scatter(point_vecs,labels,topic_vec=[],frag_means=[],frag_group=[],frag_topic=[]):
    fig, ax = plt.subplots()
#    x=[v[0] for v in point_vecs]
 #   y=[v[1] for v in point_vecs]
    pca = decomposition.PCA(n_components=2)
    pca.fit(point_vecs)
    X = pca.transform(point_vecs)
    f_x=[v[0] for v in frag_means]
    f_y=[v[1] for v in frag_means]
    x=[v[0] for v in X]
    y=[v[1] for v in X]

    color_map = {-1:'black',0: 'red',  1: 'blue',2: 'yellow',3:'grey',4:'pink',5:'cyan',6:'azure',7:'bisque',8:'gold',9:'blueviolet',10:'black',11:'navy',12:'ivory',13:'magenta',14:'brown',15:'burlywood',16:'seagreen',17:'darkorange',18:'deeppink',19:'cyan4'}
    if frag_means!=[]:
        for i in range(len(frag_means)):
            if len(frag_group[i])>0:
                circle = plt.Circle((f_x[i],f_y[i]), 0.5, color=color_map[frag_topic[i]], fill=False)
                ax.add_artist(circle)


    if topic_vec!=[]:
        ax.scatter(x,y,color=[color_map[i] for i in topic_vec])
    else:
        ax.scatter(x,y)
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i],y[i]))
    plt.show()
    
def normal_pdf(mean,x):
    #np.exp(np.array(mean)-np.array(x))
    return np.exp(-(np.linalg.norm(x-mean)**2))


def matrix_to_edges(matrix,labels,transition_matrix=[]):
    edges=[]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j]==1:
                if transition_matrix!=[]:
                    print transition_matrix[i][j]
                    #edges.append((str(i),str(j),str(round(transition_matrix[i][j],2)),{"style":"dashed"}))
                    edges.append((str(i),str(j),str(round(transition_matrix[i][j],2))))
                else:
                    #edges.append((str(i),str(j),"",{"style":"dashed"}))
                    edges.append((str(i),str(j),""))
    return edges
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
def get_index_map(v_list):
    i_map={}
    for ind,v in enumerate(v_list):
        i_map[v]=ind
    return i_map
def graph_to_matrix(G):
    node_num=len(list(G.vertices()))
    i_map=get_index_map(G.vertices())
    matrix=np.zeros(shape=(node_num,node_num))
    labels=[0]*node_num
    for e in G.edges():
        matrix[i_map[e.source()]][i_map[e.target()]]=1
    for v in G.vertices():
        labels[i_map[v]]=G.vp.labels[v]
    return matrix,labels

def get_index_label_dict(labels):
    index_label_dict={}
    for ind,la in enumerate(labels):
        index_label_dict[la]=ind
    return index_label_dict
def update_topic_masks(ecount_matrix,topic_masks,jump_prop=0.1):
    for t in range(len(ecount_matrix)):
        s=float(sum(ecount_matrix[t]))
        for l in range(len(ecount_matrix[t])):
            if ecount_matrix[t][l]==0:
                if uniform(0,1)<0.5 :
                    topic_masks[t][l]=0
                else:
                    topic_masks[t][l]=1
            else:
                topic_masks[t][l]=1


def get_distance_mat(g,labels):
    distance_matrix= shortest_distance(g,directed=False)
    d_m=[]
    #print len(labels)
    for i in  range(len(labels)):
        d_m.append(distance_matrix[i])
    return d_m

def get_doc_components(doc_labels):
    doc_num=len(set(doc_labels))
    doc_components=[]
    for i in range(doc_num):doc_components.append([])
    for i in range(len(doc_labels)):doc_components[doc_labels[i]].append(i)
    return doc_components
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

def plot_graph(matrix,labels,motif_assign_vec=[],fname='test',transition_matrix=[]):#graphvz
    G = gv.Digraph(format='png')
    styles = {
    'graph': {
        'rankdir': 'random',
    }}
    G.graph_attr.update(('graph' in styles and styles['graph']) or {})


    color_map = {-1:'black',0: 'red',  1: 'blue',2: 'yellow',3:'grey',4:'pink',5:'cyan2',6:'azure',7:'bisque1',8:'gold',9:'blueviolet',10:'black',11:'navy',12:'ivory3',13:'magenta',14:'brown',15:'burlywood',16:'seagreen',17:'darkorange',18:'deeppink',19:'cyan4'}
    shape_map = {1:"box",0:"circle"}
    if len(motif_assign_vec)==0:
        add_nodes(G,[(str(i),{'label':labels[i],'color':'black'}) for i in range(len(labels))])
    else:
        add_nodes(G,[(str(i),{'label':labels[i],'color':color_map[motif_assign_vec[i]]}) for i in range(len(labels))])
    add_edges(G,matrix_to_edges(matrix,labels,transition_matrix=transition_matrix))
    G.render(fname)
def get_distance_matrices(matrix,labels):
    components=get_connected_components(matrix,labels)
    dis_matrices=[]
    for c in range(len(components)):
        component_matrix,component_labels=extract_matrix(matrix,labels,components[c])
        g,lp=matrix_to_graph(component_matrix,component_labels)
        d_m=get_distance_mat(g,component_labels)
        dis_matrices.append(d_m)
    return dis_matrices,components

def get_node_vectors(matrix,labels,dimension=5):
    components=get_connected_components(matrix,labels)
    point_vecs=[]
    for c in range(len(components)):

        component_matrix,component_labels=extract_matrix(matrix,labels,components[c])
        g,lp=matrix_to_graph(component_matrix,component_labels)
        d_m=get_distance_mat(g,component_labels)
        mds = MDS(n_components=dimension,max_iter=1000,dissimilarity='precomputed',verbose=1)
        point_vec= mds.fit_transform(d_m)
        point_vecs.append(point_vec)
    return point_vecs,components

def get_connected_components(matrix,labels):
    g,lp=matrix_to_graph(matrix,labels)
    lc=label_components(g,directed=False)
    cp=lc[0]
    connected_components=[]
    for i in range(len(lc[1])):connected_components.append([])
    for i in range(len(labels)):
        connected_components[cp[i]].append(i)
    return connected_components
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
def normalize(arr):
    s=sum(arr)
    for i in range(len(arr)):arr[i]/=float(s)



def filp_noise_node(matrix,motif_assign_vec):
    neigh_motif=None
    n=len(matrix)
    for i in range(n):
        s=0
        #if motif_assign_vec[i]==-1:continue
        cur_motif=motif_assign_vec[i]
        neigh_motif=None
        filp=True
        for j in range(n):
            if matrix[i][j]==0:continue
            s+=1
            if motif_assign_vec[j]==cur_motif:
                filp=False
                break
            if neigh_motif!=None and motif_assign_vec[j]!=neigh_motif:
                filp=False
                break
            neigh_motif=motif_assign_vec[j]
        if filp==True:
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


def delete_node_set(matrix,labels,motif_assign_vec,delete_indexes):
    new_matrix=np.delete(matrix,list(delete_indexes),0)
    new_matrix=np.delete(new_matrix,list(delete_indexes),1)
    new_labels=np.delete(labels,list(delete_indexes))
    new_motif_assign_vec=np.delete(motif_assign_vec,list(delete_indexes))
    return new_matrix,new_labels,new_motif_assign_vec
def get_int_labels(labels):
    t=list(set(labels))
    index_label_dict=get_index_label_dict(t)
    int_labels=[index_label_dict[la] for la in labels]
    return int_labels,t,index_label_dict
def matrices_to_one_matrix(matrices,labels_list):
    matrix=[]
    labels=[]
    doc_labels=[]
    for i in range(len(matrices)):
        new_matrix=deepcopy(list(matrices[i]))
        matrix_node_num=len(matrix)
        new_matrix_node_num=len(new_matrix)
 #       if len(new_matrix)!=len(workflows_labels[i]):
  #          print 'm_len',len(new_matrix),'l_len',len(workflows_labels[i])
        for j in range(matrix_node_num):
            t=[0]*new_matrix_node_num
            matrix[j].extend(t)
        
        for j in range(new_matrix_node_num):
            new_matrix[j]=([0]*matrix_node_num)+list(new_matrix[j])
        #print len(new_matrix),len(workflows_labels[i])
        matrix.extend(new_matrix)
        if labels_list!=[]:
            labels.extend(labels_list[i])
    return matrix,labels
def effective_topic_num(topics,threshold):
    '''
    s=0.0
    for i in topics:
        s+=sum(i)
    a = sorted(topics, key=lambda x:sum(x) , reverse=True)
#    print 'dsd',a
    i=0
    cs=0.0
    while cs/s<0.99:
        cs+=sum(a[i])
        i+=1
    '''
    topic_num=len(topics)

    for i in topics:
        if sum(i)<threshold:
            topic_num-=1

    return topic_num
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
#    for i in range(motif_num):
 #       matrix_list[i],labels_list[i]=matrices_to_one_matrix(matrix_list[i],labels_list[i])
    return matrix_list,labels_list
def get_workflow_list(matrix,labels):
    matrix_list=[]#each entry is matrix with same topic
    labels_list=[]
    connected_components=get_connected_components(matrix,labels)

    for c in connected_components:
        m,l=extract_matrix(matrix,labels,c)
        matrix_list.append(m)
        labels_list.append(l)
    return matrix_list,labels_list

def overlap_sim(l1,l2):
    m_n=0
    not_search=[]
    for j in range(len(l2)):not_search.append(1)
    for i in range(len(l1)):
        for j in range(len(l2)):
            if not_search[j]==0:continue
            if l1[i]==l2[j]:
                m_n+=1
                not_search[j]=0
                break
    return float(m_n)/max(len(l1),len(l2))
def pick_representative(matrix_list,labels_list):
    best_ind=0
    frag_num=len(matrix_list)
    score_array=[0]*frag_num
    for i in range(frag_num):
        for j in range(i+1,frag_num):
            s=overlap_sim(labels_list[i],labels_list[j])
            score_array[i]+=s
            score_array[j]+=s
    best_ind=np.argmax(score_array)

    return matrix_list[best_ind],labels_list[best_ind]

def extract_representatives(matrix,labels,topic_assign_vec,topic_num):
    matrix_list,labels_list= group_by_topic(matrix,labels,topic_assign_vec,topic_num)

    #matrix,labels=matrices_to_one_matrix(matrix_list,labels_list)
    #return matrix,labels
    repre_matrices=[]
    repre_labels=[]
    for t in range(topic_num):
        m,l=pick_representative(matrix_list[t],labels_list[t])
        repre_matrices.append(m)
        repre_labels.append(l)

    matrix,labels=matrices_to_one_matrix(repre_matrices,repre_labels)
   # return matrix_list[0],labels_list[0]
    return matrix,labels
def matrix_to_graph(matrix,labels):
    g=Graph(directed=True)
    node_num=len(matrix)
    #print len(matrix),len(labels)
    g.add_vertex(node_num)
    label_property= g.new_vertex_property("string")  
    for i in range(node_num):
        label_property[g.vertex(i)]=labels[i]
    for i in range(node_num):
        for j in range(node_num):
            if matrix[i][j]==1:
                g.add_edge(g.vertex(i),g.vertex(j))
    return g,label_property
def fragment_filter(frag_freq_array,frag_matrices,frag_labels_list):
    delete_indexes=[]
    for i in range(len(frag_freq_array)):
        if frag_freq_array[i]<2:
            delete_indexes.append(i)
    return np.delete(frag_freq_array,list(delete_indexes),0).tolist(),np.delete(frag_matrices,list(delete_indexes),0).tolist(),np.delete(frag_labels_list,list(delete_indexes),0).tolist()

def sort_frag_by_count(matrix,labels,cor_matrix,cor_labels,filter_frag=False):

    frag_matrices,frag_labels_list=get_workflow_list(matrix,labels)
    cor_matrices,cor_labels_list=get_workflow_list(cor_matrix,cor_labels)
    frag_graphs=[]
    cor_graphs=[]


    for i in range(len(frag_matrices)):
        g,lp=matrix_to_graph(frag_matrices[i],frag_labels_list[i])
        frag_graphs.append([g,lp])
    for i in range(len(cor_matrices)):
        g,lp=matrix_to_graph(cor_matrices[i],cor_labels_list[i])
        cor_graphs.append([g,lp])
    frag_freq_array=[0]*len(frag_matrices)

    for i in range(len(frag_matrices)):
        for j in range(len(cor_matrices)):
            if len(frag_matrices[i])>len(cor_matrices[j]):continue
            if len(frag_matrices[i])>70 and len(cor_matrices[j])>100:continue
            
            if subgraph_isomorphism(frag_graphs[i][0],cor_graphs[j][0],vertex_label=(frag_graphs[i][1],cor_graphs[j][1]),max_n=2)!=[]:
                frag_freq_array[i]+=1
                continue
  #      print i,frag_freq_array[i]

    if filter_frag==True:
 #       print "number of frag",len(frag_matrices)
        frag_freq_array,frag_matrices,frag_labels_list=fragment_filter(frag_freq_array,frag_matrices,frag_labels_list)
#        print "number of filter frag",len(frag_matrices)
    a = sorted(zip(frag_freq_array,frag_matrices,frag_labels_list), key=lambda x:x[0] , reverse=True)
    sorted_frag_mas=[]
    sorted_frag_las=[]
    sorted_frag_freq=[]
    for x in a:
        f,m,l=x
        sorted_frag_mas.append(m)
        sorted_frag_las.append(l)
        sorted_frag_freq.append(f)
#    print 'sss',sorted_frag_mas[1],sorted_frag_las[1]
    return sorted_frag_mas,sorted_frag_las,sorted_frag_freq

def matrix_corpus_to_graph_corpus(matrix,labels):
    G=[]
    connected_components=get_connected_components(matrix,labels)
    for c in connected_components:
        component_matrix,component_labels=extract_matrix(matrix,labels,c)
        g,lp=    matrix_to_graph(component_matrix,component_labels)
        label_property= g.new_vertex_property("string")  

        for i in range(len(component_matrix)):
            label_property[i]=component_labels[i]
        g.vp.labels=label_property
        G.append(g)

    return G
