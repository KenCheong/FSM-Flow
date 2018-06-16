# PSM-Flow:Probabilistic Subgraph Mining for Workflow
**PSM-Flow** is an algorithm for mining frequent fragments in workflow data set.
### Software Prerequisites
Python Libraries include-

1.sklearn

2.graph_tool

3.graphviz

4.numpy

### How to run

```
$ python test.py 
```
The user can run test.py to test runtime of PSM-Flow on data generated by sythetic_graph_generator.py. The output statistic is stored in PSM_stat.csv file.

Three workflow data from LONI-pipeline system are stored in Loni_workflows folder.

The Gibb sampling procedure is implemented in PSM_Flow.py file. 
sythetic_graph_generater.py was implemented to generate toy dataset for testing purposes.


