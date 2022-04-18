# SPCUP AGH ACOUSTIC TEAM
## ENV CONFIG

Requirements: miniconda or anaconda, linux-64

'conda create --name myenv --file spec-file.txt'
'conda activate myenv'

You also need to install manually:
'pip install python_speech_features'

## Datasets

Please put the challenge data in the 'datasets' directory or provide your own paths in 'data_settings.yml' file.

## Evaluation instruction

1. Specify the paths to pretrained model, eval part1 and part2 data folders in 'data_settings.yml' in evaluation section. 
2. Run 'python eval.py' script.

## Training instruction

1. Specify the paths to folders with data in 'data_settings.yml' in training section. 
2. Run 'bash run_augmentation.sh' (Augmentation was performed on windows)
3. Run 'python train.py'