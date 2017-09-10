
from utility import *
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


