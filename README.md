## Python implementation of RECAPP and Catalyst for finite sum problems

This repository contains the code to reproduce the experiments from the paper 
"RECAPP: Crafting a More Efficient Catalyst for Convex Optimization" by 
Yair Carmon, Arun Jambulapati, Yujia Jin and Aaron Sidford.

## Dependencies 
To create a conda environment (called `recapp`) run: 
```  
conda env create -f environment.yml  
``` 

## Running experiments 
For examples and explanations on how to run the code, see the notebook:
```  
example.ipynb  
``` 
For the (automatically generated) command line interface explanation, run:
```
python experiment.py algname --help
```
where `algname` is either `svrg`, `catalyst`, or `recapp`.

## Reference
```bibtex
@inproceedings{carmon2022recapp,
	title={{RECAPP}: Crafting a More Efficient Catalyst for Convex Optimization}}, 
	author={Carmon, Yair and Jambulapati, Arun and Jin, Yujia and Sidford, Aaron},
	booktitle={International Conference on Machine Learning},
	year={2022}
}
```



