from typing import Final
import glob, os, subprocess, sys


DATASET_DIR: Final = os.path.join('data', 'dataset')
OUTPUT_DIR: Final = os.path.join('data', 'output')

for filepath in glob.glob(os.path.join(DATASET_DIR, '*.mp4')): #TODO ['data\\dataset\\video - iacs-camera-01 - 2024.06.10 15.59.mp4']
    root = os.path.splitext(os.path.basename(filepath))[0]
    subprocess.run([sys.executable, '-m', 'licenceplate',
        '--server', 'some_address',
        '--video', filepath,
        '--name', root,
        '--output', os.path.join(OUTPUT_DIR, f'{root}.avi'),
    ])
