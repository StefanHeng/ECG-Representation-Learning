# ECG-Representation-Learning
Self-supervised pre-training for symbolic ECG representation with inspiration from NLP

Taking inspiration from recent advancements in NLP, including vector embedding, 
attention and self-supervised pre-training, and taking advantage of large amount of 
ECG data generated, pre-train a model to understand ECG signals. 


## The combined dataset 
TODO 

## To use 
Have the datasets linked above installed. 

Modify the file [`data_path.py`](https://github.com/StefanHeng/ECG-Representation-Learning/blob/master/data_path.py) 
in root level.


In the file specify the following variables with
your system data path, and relative repository & dataset folder names, and
an `OS` variable
as shown below, `/` will be replaced by `\` for 'Windows'.
```python
PATH_BASE = '/Users/stefanh/Documents/UMich/Research/ECG-Classify'  # Absolute system path for root directory 
DIR_PROJ = 'ECG-Representation-Learning'  # Repo root folder name
DIR_DSET = 'datasets'  # Dataset root folder name
OS = 'Mac'  # 'Windows' for Windows
``` 


Also, a `datasets` folder should be kept at the same level as
this repository, with dataset folder names specified as
in [`config.json`](https://github.com/StefanHeng/Symbolic-Music-Generation/blob/master/config.json).

