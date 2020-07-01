# VIDHOP
VIDHOP is a virus host predicting tool. Its able to predict influenza A virus, rabies lyssavirus and rotavirus A.

## Install: ##

We recommend to use miniconda for the enviroment management

1.  [Download and install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

2.  Create a Conda environment with a Python 3.6 or higher

    ```bash
    conda create -n vidhop python=3.7
    ```
    
3.  Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use epidope.

    ```bash
    conda activate vidhop
    ```
4. Install vidhop via conda 

    ```bash
    conda install -c flomock -c conda-forge vidhop
    ```





## How to use:
First, if have not already activated your conda environment:
```
conda activate vidhop
```
now you can simply use vidhop via:

  **Example:**
```
  $ vidhop -i /home/user/fasta/influenza.fna -v influ
```
  present only hosts which reach a threshold of 0.2
```
  $ vidhop -i /home/user/fasta/influenza.fna -v influ -t 0.2
```
  if you want the output in a file
```
  $ vidhop -i /home/user/fasta/influenza.fna -v influ -o /home/user/vidhop_result.txt
```
  use multiple fasta-files in directory
```
  $ vidhop -i /home/user/fasta/ -v rabies
```  
  use multiple fasta-files in directory and only present top 3 host predictions per sequence
```
  $ vidhop -i /home/user/fasta/ -v rabies -n_hosts
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

<br><br>
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
