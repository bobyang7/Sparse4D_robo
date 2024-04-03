import json
import os

# please specify your path here
folder = "/home/bo.yang5/other/Sparse4D-full2/test/sparse4dv3_temporal_r50_1x8_bs6_256x704_robo"
files = os.listdir(folder)

all_dict = {}
for file in files:
    path = os.path.join(folder, file, "img_bbox", "results_nusc.json")
    with open(path, "r") as f:
        data = json.load(f)
    all_dict[file] = data

with open("/home/bo.yang5/other/Sparse4D-full2/pred.json", "w") as f:
    json.dump(all_dict, f)
    a = 1
