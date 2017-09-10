import copy
from random import randint,uniform,shuffle

topic_num=3

topic_labels=[]                        
for i in range(topic_num):topic_labels.append([])
node_per_topic=5
for i in range(node_per_topic):topic_labels[0].append("A"+str(i))
for i in range(node_per_topic):topic_labels[1].append("B"+str(i))
for i in range(node_per_topic):topic_labels[2].append("C"+str(i))
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

topic_structure=[copy.deepcopy(empty_net[0]),copy.deepcopy(empty_net[1]),copy.deepcopy(empty_net[2])]

topic_structure[0][0][1]=1
topic_structure[0][1][2]=1
topic_structure[0][2][3]=1
topic_structure[0][3][4]=1
topic_structure[1][0][1]=1
topic_structure[1][1][2]=1
topic_structure[1][2][3]=1
topic_structure[1][2][4]=1
topic_structure[2][0][1]=1
topic_structure[2][1][2]=1
topic_structure[2][1][3]=1
topic_structure[2][1][4]=1





def get_noisy_topic(matrix,labels): 
    new_matrix=copy.deepcopy(matrix)
    new_labels=copy.deepcopy(labels)
    #filp label
    for i in range(1):
        filp_label=randint(0,len(labels)-1)
        picked_label=randint(0,len(total_topic_labels)-1)
        new_labels[int(filp_label)]=total_topic_labels[picked_label]

    return new_matrix,new_labels

def get_toy_graph(repeat_times=10):
    matrix,labels=[],[]
    subgraphs_matrices,subgraphs_labels=[],[]
    gen_order=[]
    for i in range(topic_num):
        gen_order+=[i]*repeat_times
    shuffle(gen_order)
    for i in range(len(gen_order)):

        new_matrix,new_labels=get_noisy_topic(topic_structure[gen_order[i]],topic_labels[gen_order[i]])
        subgraphs_matrices.append(copy.deepcopy(new_matrix))
        subgraphs_labels.append(copy.deepcopy(new_labels))
        matrix_node_num=len(matrix)
        new_matrix_node_num=len(new_matrix)
        for i in range(matrix_node_num):
            t=[0]*new_matrix_node_num
            matrix[i].extend(t)
        if matrix_node_num!=0:
            for i in range(1):#combine link 
                src=randint(0,matrix_node_num-1)
                dest=randint(0,new_matrix_node_num-1)
                matrix[src][matrix_node_num+dest]=1
        for i in range(new_matrix_node_num):
            new_matrix[i]=([0]*matrix_node_num)+new_matrix[i]
        matrix.extend(new_matrix)
        labels.extend(new_labels)
    return matrix,labels,subgraphs_matrices,subgraphs_labels


