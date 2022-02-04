# FORCE Hackathon 2019
Code for translating formation descriptions to transcriptions/classifications (grain size) using machine learning.

## Retrieving dataset 

Download the following Excel spreadsheet to a folder named __data__:
[RealPore Por Perm Lithology data 1240 Wells Norway public.xlsx]<https://drive.google.com/open?id=17kOO2637vXV1jADdo6i4PpYt7X6BYY4N>

If named exactly **RealPore Por Perm Lithology data 1240 Wells Norway public.xlsx**, it will be picked up automatically when running the machine_translation.py script. Otherwise, specify the path to the dataset with the **--dataset** option.

Original link seems to be dead. 

The spreadsheet is also available at <https://zenodo.org/record/4419060>

A version 2.0 of the data is available at <https://zenodo.org/record/4723018>


## Python dependencies

Tested with Python 3.9.

See requirements.txt file for further dependencies

Optionally install tensorflow-gpu instead of tensorflow package if a GPU is available,
to speed up training.

## Training formation description to transcriptions or formation descriptions to grain size

Run the machine_translation.py script to start the machine learning


Example usages:
```bash
python machine_translation.py transcription
# or
python machine_translation.py "grain size"
# or
python machine_translation.py --help # to see other options.
```
