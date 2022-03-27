# ECG-Representation-Learning
Self-supervised pre-training for ECG representation with inspiration from recent advancements 
in transformers in Natural Language Processing and Computer Vision. 


## The combined dataset 


| Name                                                                                                                                                     | # records |
|----------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| [St Petersburg INCART 12-lead Arrhythmia Database](https://www.physionet.org/content/incartdb/1.0.0/)                                                    | 75        |
| [PTB Diagnostic ECG Database](https://www.physionet.org/content/ptbdb/1.0.0/)                                                                            | 549       |
| [PTB-XL, a large publicly available electrocardiography dataset](https://physionet.org/content/ptb-xl/1.0.1/)                                            | 21,837    |
| [China Physiological Signal Challenge 2018](http://2018.icbeb.org/Challenge.html)                                                                        | 6,877     |
| CSPC extra/unused dataset                                                                                                                                | 3,453     |
| Georgia 12-lead ECG Challenge (G12EC) Database                                                                                                           | 10,344    |
| [A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients](https://figshare.com/collections/ChapmanECG/4560497/2) | 10,646    |
| [Test set](https://zenodo.org/record/3765780#.YX39IC-B1qs) from paper *Automatic diagnosis of the 12-lead ECG using a deep neural network*               | 827       |

Note that all entires apart from the last one are part of the *PhysioNet - Computing in Cardiology Challenge 2021* (CinC21). We collect the dataset from the original publishing source if available since the versions from CinC21 had records removed. 



## To use
1< Have the datasets linked above downloaded. 

2> Modify the `DIR_DSET` variable in file [`data_path.py`](https://github.com/StefanHeng/ECG-Representation-Learning/blob/master/ecg_transformer/util/data_path.py) 
as instructed. 

A folder named as `DIR_DSET` should be kept at the same level as
this repository, with dataset folder names specified as
in [`config.json`](https://github.com/StefanHeng/ECG-Representation-Learning/blob/master/util/config.json).

