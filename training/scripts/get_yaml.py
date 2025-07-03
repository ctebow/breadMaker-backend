import yaml
import os

with open("breadMaker.v1-handdrawn-circuits-2025-06-20-1-18pm.yolov8/data.yaml", "r") as f:
    data_true = yaml.safe_load(f)

with open("ref des raw.v1i.yolov8/data_old.yaml", "r") as f:
    data_new = yaml.safe_load(f)

component_mapping_true = {}
for idx, name in enumerate(data_true["names"]):
    component_mapping_true[idx] = name

component_mapping_new = {}
for idx, name in enumerate(data_new["names"]):
    component_mapping_new[idx] = name

print(component_mapping_true)
print(" ")
print(component_mapping_new)

