import pickle
import sys
from random import shuffle
import json

def text_to_graph(rawf_name):
#    rawf_name= 'Templates_Inference_Text_Analytics' 
    #rawf_name= 'Templates_No_Inference_Text_Analytics' 
    #rawf_name= 'Template_no2' 
    fobj=open(rawf_name)
    workflows=[]#list of [[7,8],[8,9]]
    new_workflows=[]#list of [[1,2],[3,4]]
    workflows_labels=[]#list of [stemmer, classifier]
    id_to_label=[]
    ###
    for row in fobj.readlines():
        if row.split(' ')[0]=='%Processing:' or row.split(' ')[0]=='t':
            workflows.append([])
            id_to_label.append({})
        else:
           # if int(row.split(' ')[1])<144:
           # print row.split(' ')[0]
            if row.split(' ')[0]=='v' and len(row.split('/'))>1:
                id_to_label[-1][row.split(' ')[1]]=row.split('/')[-1].strip('\r\n')
            elif row.split(' ')[0]=='d' or row.split(' ')[0]=='e':
                workflows[-1].append([row.split(' ')[1],row.split(' ')[2]])
    ###
    '''
    for i in id_to_label:
        print i
    '''
    ###new index space(filter unneed nodes)
    for i in range(len(workflows)):
        if len(workflows[i])==0:continue
        ind_set=set()
        new_workflows.append([])
        workflows_labels.append([])
        for j in workflows[i]:
            ind_set.add(j[0])
            ind_set.add(j[1])
        lind_set=list(ind_set)
        for j in lind_set:
            workflows_labels[-1].append(id_to_label[i][j])
        for j in workflows[i]:
            new_workflows[-1].append([lind_set.index(j[0]),lind_set.index(j[1])])
#    print len(new_workflows)
    return new_workflows,workflows_labels
  

def workflow_to_graph(workflow):
    node_num=0
    for i in workflow:
#        if i==0:
        node_num=max([node_num,i[0],i[1]])
    node_num+=1
    matrix=[]
    for i in range(node_num):
        matrix.append([0]*node_num)
    for i in workflow:
        matrix[i[0]][i[1]]=1
    return matrix

def workflows_to_one_graph(workflows,workflows_labels):
    matrix=[]
    labels=[]
    doc_labels=[]
    for i in range(len(workflows)):
        new_matrix=workflow_to_graph(workflows[i])
        matrix_node_num=len(matrix)
        new_matrix_node_num=len(new_matrix)
        for k in range(new_matrix_node_num):doc_labels.append(i)
        for j in range(matrix_node_num):
            t=[0]*new_matrix_node_num
            matrix[j].extend(t)
        
        for j in range(new_matrix_node_num):
            new_matrix[j]=([0]*matrix_node_num)+new_matrix[j]
        print len(new_matrix),len(workflows_labels[i])
        matrix.extend(new_matrix)
        labels.extend(workflows_labels[i])
    return matrix,labels,doc_labels
def save_pickle(matrix,labels,doc_labels,fname):
    with open(fname, 'wb') as handle:
        json.dump([matrix,labels,doc_labels], handle)

corpus_name=sys.argv[1]
output_dir='./output/'
corpus_dir='./corpus/'
workflows,workflows_labels=text_to_graph(corpus_dir+corpus_name)
matrix,labels,doc_labels=workflows_to_one_graph(workflows,workflows_labels)
print 'saving'
save_pickle(matrix,labels,doc_labels,output_dir+corpus_name+'.pickle')
print 'finish saving'
   

