# Textual_Pseudo_Augmentation


This open source library was created in order to generate textual pseudo instances. It aims at extracting sentences from twitter's data (mostly) which must be stored in a specific way. It can also be used for any other dataset which fullfil the below description of indices.

The implementation was done in scala for distributed environment: Apache Spark. 

All the necessery files are stored in the folder common-files. These files must be moved to the directory which the .jar will be saved. Datasets on the other hand must be stored in the HDFS (or same as common-files in case spark runs locally).

# Dataset Description

Each instance of the dataset must be stored in a new line. Each attribute of an instance has to be separated by comma character (",").
Below is the list with the indices for each attribute:

0. Document Id: Long
1. Sentiment: String. In our case we store "positive" or "negative".
2. Text: String. The sentence preprocessed (for our case)

# Input Arguments

For our evaluation we have used different datasets such as SemEval 2017, Sentiment140 and TSentiment15 (all contain tweets). After creating the .jar one should the name of the dataset and an option (in our case we use 3 different datasets, so options are 1: Sentiment140, 2: SemEval17, 3: TSentiment15)

i.e: $ ./spark-submit --class MultiEvaluation Textual_Pseudo_Augmentation.jar distant_supervision_dataset.txt 1 

