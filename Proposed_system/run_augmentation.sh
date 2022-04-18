#!/bin/bash
matlab -batch process_1;
matlab -batch process_2;
python mp3_to_wav.py;
python speed_pitch.py;