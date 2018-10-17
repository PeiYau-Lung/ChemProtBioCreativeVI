cat ./data/chemprot_training_abstracts.tsv ./data/chemprot_development_abstracts.tsv > ChemProtTrain_abstracts.tsv
cat ./data/chemprot_training_entities.tsv ./data/chemprot_development_entities.tsv > ChemProtTrain_entities.tsv
cat ./data/chemprot_training_relations.tsv ./data/chemprot_development_relations.tsv > ChemProtTrain_relations.tsv
cp ./data/chemprot_test_abstracts.tsv .
cp ./data/chemprot_test_entities.tsv .
python ./src/Ent2Relation.py -a ChemProtTrain_abstracts.tsv -e ChemProtTrain_entities.tsv -r ChemProtTrain_relations.tsv -p ChemProtTrain
python ./src/Ent2Relation.py -a chemprot_test_abstracts.tsv -e chemprot_test_entities.tsv -r chemprot_test_relations.tsv -p ChemProtTest
python ./src/Abs2Triplets.py -a ChemProtTrain_abstracts.tsv -e ChemProtTrain_entities.tsv -r ChemProtTrain_ALLrelations.txt -p ChemProtTrain
python ./src/Abs2Triplets.py -a chemprot_test_abstracts.tsv -e chemprot_test_entities.tsv -r ChemProtTest_ALLrelations.txt -p ChemProtTest
python ./src/RewriteCorpus.py -i ChemProtTrain_triplets.txt -o ChemProtTrain_sentence.txt
python ./src/RewriteCorpus.py -i ChemProtTest_triplets.txt -o ChemProtTest_sentence.txt
python ./src/RunParser.py -i ChemProtTrain_sentence.txt -p ChemProtTrain
python ./src/RunParser.py -i ChemProtTest_sentence.txt -p ChemProtTest
python ./src/Dep2Graph.py -p ChemProtTrain
python ./src/Dep2Graph.py -p ChemProtTest
python ./src/sentFromGraph.py -i ChemProtTrain_nnparsed_graph.txt -p ChemProtTrain
python ./src/sentFromGraph.py -i ChemProtTest_nnparsed_graph.txt -p ChemProtTest
python ./src/TagTriplets.py -s ChemProtTrain_sentFromGraph.txt -c ChemProtTrain_triplets.txt -p ChemProtTrain
python ./src/TagTriplets.py -s ChemProtTest_sentFromGraph.txt -c ChemProtTest_triplets.txt -p ChemProtTest
python ./src/shortestPath.py -g ChemProtTrain_nnparsed_graph.txt -t ChemProtTrain_tagged.txt -l ChemProtTrain_trip_label.txt -p ChemProtTrain
python ./src/shortestPath.py -g ChemProtTest_nnparsed_graph.txt -t ChemProtTest_tagged.txt -l ChemProtTest_trip_label.txt -p ChemProtTest
python ./src/getFeatures_pair.py -p ChemProtTrain
python ./src/getFeatures_pair.py -p ChemProtTest
python ./src/getFeatures_triplets.py -p ChemProtTrain
python ./src/getFeatures_triplets.py -p ChemProtTest
python ./src/L1_Model.py -r ChemProtTrain_Features_pair.txt -s ChemProtTest_Features_pair.txt -t 0 -a 0
python ./src/L1_Model.py -r ChemProtTrain_Features_triplet.txt -s ChemProtTest_Features_triplet.txt -t 1 -a 0
python ./src/L2_Model.py -r ChemProtTrain_Features_pair.txt -s ChemProtTest_Features_pair.txt -t 0 -l 1 -a 0
python ./src/L2_Model.py -r ChemProtTrain_Features_triplet.txt -s ChemProtTest_Features_triplet.txt -t 1 -l 1 -a 0
python ./src/L3_Model.py -r ChemProtTrain -s ChemProtTest
