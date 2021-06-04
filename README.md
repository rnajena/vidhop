# VIDHOP
VIDHOP is a virus host predicting tool. Its able to [predict](#examples-for-vidhop-predict:) influenza A virus, rabies lyssavirus and rotavirus A. Furthermore the user can [train](#train-and-use-your-own-model:) its own models for other viruses and use them with VIDHOP.

## Install: ##

We recommend to use linux and miniconda for the enviroment management

1.  [Download and install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

2.  Create a Conda environment with a Python 3.7

    ```bash
    conda create -n vidhop python=3.7
    ```
    
3.  Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use VIDHOP.

    ```bash
    conda activate vidhop
    ```
4. Install vidhop via conda 

    ```bash
    conda install -c flomock -c anaconda vidhop=0.9
    ```

## How to use:

VIDHOP has three different commands each with its own parameter set.

**make_dataset:** create the data structure needed for training

**training:** train a model on your training files generated with make_dataset

**predict:** predict the host of the viral sequence given 

Use ```vidhop --help``` to see this summary of all three methods.

Likely you'll mainly use predict, see below various examples: 

#### examples for vidhop predict:


```
vidhop predict -i /home/user/fasta/influenza.fna -v influ
```
  present only hosts which reach a threshold of 0.2
```
vidhop predict -i /home/user/fasta/influenza.fna -v influ -t 0.2
```
  if you want the output in a file
```
vidhop predict -i /home/user/fasta/influenza.fna -v influ -o /home/user/vidhop_result.txt
```
  use multiple fasta-files in directory
```
vidhop predict -i /home/user/fasta/ -v rabies
```  
  use multiple fasta-files in directory and only present top 3 host predictions per sequence
```
vidhop predict -i /home/user/fasta/ -v rabies -n_hosts
```  
Use your own trained models generated with [vidhop training](#Train-and-use-your-own-model:). 
You need to specify the path to the .model file you want to use. It's located in the output directory of vidhop 
training. You can choose between the model with the lowest loss and the model with the highest accuracy while training.
```
vidhop predict -v /home/user/out_training/model_best_acc_testname.model -i /home/user/fasta/my_favourite_virus.fna
```

**Options:**

command | what it does
  ------------- | -------------
-i, --input     |either raw sequences or path to fasta file or directory with multiple files.  [required]
-v, --virus     |select virus species (influ, rabies, rota) [required]
-o, --outpath   |path where to save the output
-n, --n_hosts   |show only -n most likely hosts
-t, --thresh    |show only hosts with higher likeliness then --thresh
--auto_filter   |automatically filters output to present most relevant host
--help          |show this message and exit.
--version       |show version number from vidhop


## Train and use your own model:

**If you like to skip ahead see [toy-example](#toy-example).**

Train your own model for other viruses than the provided ones (influenza A virus, rabies lyssavirus and rotavirus A) is simple.
The workflow consists of three steps:

1.  [vidhop make_dataset](#vidhop-make_dataset)

2.  [vidhop training](#vidhop-training)

3.  [vidhop predict](#examples-for-vidhop-predict)

 

#### vidhop make_dataset

To generate the data sets needed for training you need to provide two input files.

1.  A sequence file containing in each line a DNA sequence.

2.  A host file containing the name of the host corresponding to the DNA sequence at identical line number in the sequence file.

**example input**

sequences.txt | hosts.txt
  ------------- | -------------
AAATTT | human
CGTATA | swine
CGTATT | swine

**examples for vidhop make_dataset:**

Example:
set input and output parameter
```
vidhop make_dataset -x /home/user/input/seq.txt -y /home/user/input/host.txt -o /home/user/trainingdata/
```

change the validation set size and provide datastructure for repeated undersampling
```
vidhop make_dataset -x /home/user/input/seq.txt -y /home/user/input/host.txt -v 0.1 -r
```

command | what it does
  ------------- | -------------
-x, --sequences     |Path to the file containing sequence list  [required].
-y, --hosts     |Path to the file containing corresponding host list [required].
-o, --outpath   |Path where to save the output.
-n, --n_hosts   |Show only -n most likely hosts.
-v, --val_split_size    |Select the portion of the data which is used for the validation set.
-t, --test_split_size   |Select the portion of the data which is used for the test set.
-r, --repeated_undersampling       |Generate training files needed for reapeted undersampling while training.
--help          |Show this message and exit.

#### vidhop training

The training of a model is done by providing the output directory of vidhop make_dataset as the input of vidhop training.
The user can specify various parameter which change the architecture, training duration, input handling and further more.
For further details to different parameters like --extention_variant or --repeated_undersampling see the paper, **virus host prediction with deep learning**. 

examples for vidhop training:

set input output and name of the model
```
vidhop training -i /home/user/trainingdata/ -o /home/user/model/ --name test
```

use the LSTM archtecture and the extention variant random repeat
```
vidhop training -i /home/user/trainingdata/ --architecture 0 --extention_variant 2
```

use repeated undersampling for training, note that for this the dataset must have been created with repeated undersampling enabled
```
vidhop training -i /home/user/trainingdata/ -r
```

train the model for 40 epochs, stop training if for 2 epochs the accuracy did not increase
```
vidhop train_new_model -i /home/user/trainingdata/ --epochs 40 --early_stopping
```

command | what it does
  ------------- | -------------
-i, --inpath     |Path to the dir with training files, generated with make_dataset  [required].
-o, --outpath   |Path where to save the output.
-n, --name   |Suffix added to output file names.
-e, --epochs    |Maximum number of epochs used for training the model.
-a, --architecture   |Select architecture (0:LSTM, 1:CNN+LSTM).
-v, --extention_variant   |Select extension variant (0:Normal repeat, 1:Normal repeat with gaps, 2:Random repeat, 3:Random repeat with gaps, 4:Append gaps, 5:Smallest, 6:Online).
-s, --early_stopping   |Stop training when model accuracy did not improve over time, patience 5% of max epochs.
-r, --repeated_undersampling       |Use repeated undersampling while training, to be usable the training files must be generated with make_datasets and activated reapeted undersampling parameter.
--help          |Show this message and exit.

### toy-example

Download the test files X.txt (containing all sequences) und Y.txt (containing all corresponding hosts). 
```
wget https://github.com/flomock/vidhop/blob/master/X.txt
wget https://github.com/flomock/vidhop/blob/master/Y.txt
```

Now we prepare the dataset. As an example we define the size of the validation-set to 10%  and the test-set to 20%
 of the full dataset. Note that the all data sets will be balanced according to their host classes. To use all samples, 
 even from an unbalanced dataset, without biasing towards the most common host class, use the --repeated_undersampling 
 parameter. This effects the samples used while training. The validation and test sets will be unchanged.
  
```
vidhop make_dataset -x X.txt -y Y.txt -r -o ./make_dataset_out -v 0.1 -t 0.2
```
If a host in your dataset is bellow the recommended minimal count of 100 samples, vidhop make_dataset will return a 
warning. The expected console output:

```
warning number samples for host Artibeus lituratus low, only 90 samples
warning number samples for host Lasiurus borealis low, only 96 samples
warning number samples for host Cerdocyon thous low, only 82 samples
```

Now we train a model using the standard parameter. As input we provide the output directory of vidhop make_dataset. 
Furthermore we name our model "test_standard". 
To limit training time we use -e to limit the number of epochs to two.  
 
```
vidhop training -i ./make_dataset_out -n test_standard -e 2 -o ./trained_models
```

The output printed to the console provides information about the input provided, the current architecture used, the 
current status of the training. When the actual training is completed two models are saved. One model which represents 
the model with the lowest loss during training and one model with the highest accuracy during training, both calculated 
on the validation set. 
After training and saving these models each model is tested on the test dataset. The results are printed in the console.

Now we are able to predict the host of new sequences. For this we use [vidhop predict](#examples-for-vidhop-predict:).
You can provide either a fasta file for prediction or a DNA sequence directly. Here we use an DNA sequence. Furthermore 
we define a the virus to use the path to one of our trained models, either the one with the lowest loss or the one with 
the highest accuracy while training. If you are not sure which of the both models to use, we experienced the best 
results working with the model with the highest accuracy.

```
vidhop predict -v ./trained_models/model_best_acc_test_standard.model -o ./predictions/first_test.txt -i AAATGCTCTGAATTCGACATGAAAAAAACAAGCAACACCACTGATAAGATGAACTTTCTACGCAAGAAATGCTCTGAATTCGACATGAAAAAAACAAGCAACACCACTGATAAGATGAACTTTCTACGCAAG
```

This results in a prediction similar to:
```
>user command line input
all hosts
Lasiurus borealis: 0.061880290508270264
Procyon lotor: 0.06131688505411148
Desmodus rotundus: 0.06110725179314613
Mephitis mephitis: 0.060785286128520966
Vulpes vulpes: 0.060404110699892044
Capra hircus: 0.06007476523518562
Vulpes lagopus: 0.059015918523073196
Artibeus lituratus: 0.0590040422976017
Tadarida brasiliensis: 0.05887320265173912
Nyctereutes procyonoides: 0.0584951676428318
Eptesicus fuscus: 0.05815460532903671
Felis catus: 0.05793345347046852
Equus caballus: 0.0575258694589138
Homo sapiens: 0.05704779922962189
Bos taurus: 0.05660048499703407
Canis lupus: 0.056392643600702286
Cerdocyon thous: 0.05538821220397949
``` 
(note that the input sequence is more or less random, so don't expect a very meaningful prediction)


Thanks for using VIDHOP.

<br><br>
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
