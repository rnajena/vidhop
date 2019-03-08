# VIDHOP
VIDHOP is a virus host predicting tool. Its able to predict influenza A virus, rabies lyssavirus and rotavirus A.

**Install via:**

We recommend to use miniconda for the enviroment management
see https://conda.io/en/latest/miniconda.html

after miniconda is installed:

download the enviroment.yml file from this githubpage
```
wget https://raw.githubusercontent.com/flomock/vidhop/master/environment.yml
```
create conda enviroment from file, you may need to change the file path for environment.yml
```
conda env create -f environment.yml
```
now join the new enviroment
```
conda activate vidhop
```
install vidhop
```
pip install git+https://github.com/flomock/vidhop.git@master
```





## How to use:

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
-i, --input     |Path to fasta, csv, or directory with multiple files.  [required]
-v, --virus     |Length of the resulting word.
-o, --outpath   |Length of the analyzed tuples.
-n, --n_hosts   |Necessary if path contains fasta-file which should be used.
-t, --thresh    |Uses also all subdirectories.
--auto_filter   |Filter out microsatellites;
--help          |Show this message and exit.
--version       |show version number from vidhop

<br><br>
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
