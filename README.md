# HRD200
This GitHub repository contains the training and validation code used in "Gene expression signature for predicting homologous recombination deficiency in triple-negative breast cancer" published in <i>npj Breast Cancer</i> (link: https://doi.org/10.1038/s41523-024-00671-1)
## Preludes
### Creating a conda environment
It is recommended that users create a conda environment to run HRD200 in. <br>
```
conda create -n hrd python=3.9
conda activate hrd
...
conda deactivate
```
### Output folder format
HRD200 produces 5 sets of models `(s1 - s5)` and each set contains 5 random seeds `(rs1 - rs5)`. <br>
The output folder for each run (to store the trained models and other outputs) are as follows:
```
├── output
│   ├── s1
│   │   ├── rs1
│   │   ├── rs2
│   │   ├── rs3
│   │   ├── rs4
│   │   └── rs5
│   ├── s2
│   │   ├── rs1
│       ...
│   ...
```
## Training
Run `main.py` using the sample data provided in `data/training.csv` <br>
```
python3 main.py name_of_run data/training.csv data/labels.csv
```
## Validation
After training, run `validation.py` on the sample data provided in `data/validation.csv` <br>
The `truth_file` is optional; HRD200 will generate metrics if it exists.
```
python3 validation.py data/validation.csv output/ [data/truth.csv]
```
## Packages and dependencies
The model was trained on Python 3.9, and other dependencies can be found in the manuscript.
