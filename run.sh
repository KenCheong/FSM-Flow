#!/bin/bash
corpus='WC2.g.txt'
echo $corpus
if [ ! -f './output/'$corpus'.pickle' ]
then
    python text_to_graph.py $corpus
fi
echo 'run LDA'
python Frag_LDA.py $corpus'.pickle'

