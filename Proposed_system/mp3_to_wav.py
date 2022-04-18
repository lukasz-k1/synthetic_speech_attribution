from pydub import AudioSegment
import csv
import os
from os.path import join
import yaml
from tqdm import tqdm

# yaml read
files_path = ''
destination_path = ''

with open('data_settings.yml') as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)

    destination_path = os.path.join(settings['new_data_path'],'train_data').replace('\\','/')


files = os.listdir(destination_path)


for file in tqdm(files, total=len(files)):
    if file[-4:]=='.mp3':
        src_filepath = join(destination_path,file).replace('\\','/')
        new_filename = "{}{}".format(file[:-4],".wav")
        dst_filepath = join(destination_path,new_filename).replace('\\','/')
        
        sound = AudioSegment.from_mp3(src_filepath)
        new_file = sound.export(dst_filepath, format="wav")
        new_file.close()
        os.remove(src_filepath)