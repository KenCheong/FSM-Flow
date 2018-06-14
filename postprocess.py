
from utility import *
def overlap_num(labels1,labels2,comp1,comp2):
    l1=[]
    l2=[]
    for c in comp1:l1.append(labels1[c])
    for c in comp2:l2.append(labels2[c])
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
    return float(m_n)

def extract_fragments_by_clusters(components,frag_group,matrix,labels,topic_assign_vec):
#    filp_noise_node(matrix,topic_assign_vec)
    matrix_list=[]
    label_list=[]
    topics=[]
    topic_list=[]
    for i in range(len(frag_group)):
        for j in range(len(frag_group[i])):
            if len(frag_group[i][j])<2:continue
            component_matrix,component_labels=extract_matrix(matrix,labels,[components[i][group_node] for group_node in frag_group[i][j]])
            matrix_list.append(component_matrix)
            label_list.append(component_labels)
            topic_list.append([topic_assign_vec[components[i][frag_group[i][j][0]]]]*len(frag_group[i][j]) )

    #eliminate repeat
    delete_index=[0]*len(matrix_list)
    for i in range(len(matrix_list)):
        if delete_index[i]==1:continue
        ilen=len(matrix_list[i])
        comp=range(ilen)
        for j in range(i+1,len(matrix_list)):
            if ilen!=len(matrix_list[j]):continue
            if overlap_num(label_list[i],label_list[j],comp,comp)==ilen:
                delete_index[j]=1
    new_matrix_list=[]
    new_label_list=[]
    for i in range(len(matrix_list)):
        if delete_index[i]==1:continue
        new_matrix_list.append(matrix_list[i])
        new_label_list.append(label_list[i])
        topics.extend(topic_list[i])
    frag_matrix,frag_labels=matrices_to_one_matrix(new_matrix_list,new_label_list)
    ##remove one step frag
    connected_components=get_connected_components(frag_matrix,frag_labels)
    delete_indexes=set()
    for c in connected_components:
        if len(c)==1:
            delete_indexes.add(c[0])
    frag_matrix,frag_labels,topics=delete_node_set(frag_matrix,frag_labels,topics,delete_indexes)


    return frag_matrix,frag_labels,topics

def extract_fragments(matrix,labels,topic_assign_vec):
    ## get doc_graphs for count occurances
    cor_g,cor_lp=matrix_to_graph(matrix,labels)
    doc_graphs=[]
    connected_components=get_connected_components(matrix,labels)
    for c in connected_components:
        doc_graphs.append(extract_graph(matrix,labels,c))
    ##

    ##partition graph,remove one step,repeated frag
    filp_noise_node(matrix,topic_assign_vec)
    remove_uncontiguous_link(matrix,topic_assign_vec)
    ##find connected components
    connected_components=get_connected_components(matrix,labels)

    ##remove one step frag
    delete_indexes=set()
    for c in connected_components:
        if len(c)==1:
            delete_indexes.add(c[0])
    matrix,labels,topic_assign_vec=delete_node_set(matrix,labels,topic_assign_vec,delete_indexes)

    ##
    connected_components=get_connected_components(matrix,labels)
    ###eliminate repeat frag 
    remain_frag_indicator=[1]*len(connected_components)
    delete_frag_indexes=[]
    connected_component_grpahs=[]
    connected_component_lp=[]
    for i in range(len(connected_components)):
        g,lp=extract_graph(matrix,labels,connected_components[i])
        connected_component_grpahs.append(g)
        connected_component_lp.append(lp)

    co=0
#    print 'cc',len(connected_components)
    ##get removed ones
    for i in range(len(connected_components)):
        if remain_frag_indicator[i]==0:continue
    

        co=0
        for j in range(0,len(connected_components)):
            if i==j or remain_frag_indicator[i]==0 or remain_frag_indicator[j]==0:continue
            if len(connected_components[i])!=len(connected_components[j]):
                continue
            if j>i and subgraph_isomorphism(connected_component_grpahs[i],connected_component_grpahs[j],vertex_label=(connected_component_lp[i],connected_component_lp[j]),max_n=1) !=[]:
                remain_frag_indicator[j]=0
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
    for i in range(len(connected_components)):
        for j in range(len(connected_components[i])):
            connected_components[i][j]=new_indexes_map[connected_components[i][j]]
  ##
    connected_components=np.delete(connected_components,list(delete_frag_indexes))
    ##

    matrix,labels,topic_assign_vec=delete_node_set(matrix,labels,topic_assign_vec,delete_indexes)

#    print 'number of frag:',np.count_nonzero(remain_frag_indicator)
    ###
  

    return matrix,labels,topic_assign_vec

def extract_fragments_with_count(matrix,labels,topic_assign_vec):
    ## get doc_graphs for count occurances
    cor_g,cor_lp=matrix_to_graph(matrix,labels)
    doc_graphs=[]
    sort_count_list=[]
    connected_components=get_connected_components(matrix,labels)
    for c in connected_components:
        doc_graphs.append(extract_graph(matrix,labels,c))
    ##

    ##partition graph,remove one step,repeated frag
    filp_noise_node(matrix,topic_assign_vec)
    remove_uncontiguous_link(matrix,topic_assign_vec)
    ##find connected components
    connected_components=get_connected_components(matrix,labels)

    ##remove one step frag
    delete_indexes=set()
    for c in connected_components:
        if len(c)==1:
            delete_indexes.add(c[0])
    matrix,labels,topic_assign_vec=delete_node_set(matrix,labels,topic_assign_vec,delete_indexes)

    ##
    connected_components=get_connected_components(matrix,labels)
    ###eliminate repeat frag 
    remain_frag_indicator=[1]*len(connected_components)
    delete_frag_indexes=[]
    connected_component_grpahs=[]
    connected_component_lp=[]
    component_labels=[]
    component_ma=[]
    for i in range(len(connected_components)):
        g,lp=extract_graph(matrix,labels,connected_components[i])
        ma,la=extract_matrix(matrix,labels,connected_components[i])
        connected_component_grpahs.append(g)
        connected_component_lp.append(lp)
        component_labels.append(la)
        component_ma.append(ma)

    co=0
#    print 'cc',len(connected_components)
    ##get removed ones
    frag_ma_list=[]
    frag_la_list=[]
    for i in range(len(connected_components)):
        if remain_frag_indicator[i]==0:continue
        sort_count_list.append(0)
        frag_ma_list.append(component_ma[i])
        frag_la_list.append(component_labels[i])
    

        co=0
        for j in range(0,len(connected_components)):
            sort_count_list[-1]+=overlap_sim(component_labels[i],component_labels[j])
            if i==j or remain_frag_indicator[i]==0 or remain_frag_indicator[j]==0:continue
            if len(connected_components[i])!=len(connected_components[j]):
                continue
            if j>i and subgraph_isomorphism(connected_component_grpahs[i],connected_component_grpahs[j],vertex_label=(connected_component_lp[i],connected_component_lp[j]),max_n=1) !=[]:
                remain_frag_indicator[j]=0
                delete_frag_indexes.append(j)
      



#    print 'number of frag:',np.count_nonzero(remain_frag_indicator)
    ###


    return frag_ma_list,frag_la_list,sort_count_list


