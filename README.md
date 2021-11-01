# [`Selective Probabilistic Classifier Based on Hypothesis Testing (EUVIP 2021)`](https://ieeexplore.ieee.org/document/9483967)


## Requirements
- TensorFlow Probability 0.8.0
- Python and other general packages

## Usage
- Create a data folder with `train` and `test` subfolders, and prepare the dataset. (Note that the code uses 224 x 224 x 3 images)
- Run the command:
```
python normal_resnet.py
```
or
```
python probability_resnet.py
```
- Network output is available in `Results` folder. Analyze it with proper tools.

## Citation
```
@InProceedings{BakhshiGermi2021,
    author={Bakhshi Germi, Saeed and Rahtu, Esa and Huttunen, Heikki},
    booktitle={2021 9th European Workshop on Visual Information Processing (EUVIP)}, 
    title={Selective Probabilistic Classifier Based on Hypothesis Testing}, 
    year={2021},
    doi={10.1109/EUVIP50544.2021.9483967}}
```
