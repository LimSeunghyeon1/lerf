import json
from copy import deepcopy
import os

with open("/home/tidy/lerf/lerf/data/transforms.json", "r") as f:
    json_dict = json.load(f)
base_path = '/home/tidy/lerf/lerf/data'
frames = json_dict['frames']
new_frames = []
for i, frame in enumerate(frames):
    file_path = frame['file_path']
    file_path = '/'.join(file_path.split('/')[1:])
    file_path = os.path.join(base_path, file_path)
    if os.path.isfile(file_path):
        new_frames.append(frame)
    else:
        print("remove file path", file_path)
        
    
json_dict['frames'] = new_frames

with open("/home/tidy/lerf/lerf/data/transforms_new.json", "w") as f:
    json.dump(json_dict,f, indent=2)
