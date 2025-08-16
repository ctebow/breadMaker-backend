"""
For using datasets with different class labels, method to merge labels. 
"""

import os

mapping1 = {0: 62, 1: 14, 2: 15, 3: 18, 4: 17, 5: 18, 6: 19, 12: 20, 7: 21, 8: 22, 9: 23, 10: 24, 11: 25, 13: 26, 17: 26, 14: 27, 15: 28, 16: 29, 18: 31, 19: 32, 20: 33, 21: 34, 22: 35, 23: 36, 24: 37, 25: 38, 26: 39, 27: 40, 29: 41, 28: 42, 30: 42, 31: 43, 32: 44, 33: 45, 34: 46, 35: 47, 36: 48, 37: 49, 38: 50, 39: 51, 40: 52, 41: 53, 42: 55, 43: 56, 44: 61, 45: 57, 46: 61, 47: 62, 48: 63, 49: 64, 50: 65}

bad_ids_1 = [4, 29, 30] # "Junction", "junctions", "junction" --> Current annotations have an incorrect definition
                        # for junctions according to LTSpice netfile. 

file_path = "ref des raw.v1i.yolov8/valid/labels"

def update_classes(label_directory, mapping):
    """
    Update label .txt files with correct class id.
    """

    for filename in os.listdir(label_directory):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(label_directory, filename)
        lines = []

        with open(path, "r") as f:
            for line in f:

                parts = line.strip().split()
                id_old = int(parts[0])

                if id_old in mapping:
                    id_new = mapping[id_old]
                    parts[0] = str(id_new)
                    lines.append(" ".join(parts))

                else:
                    print(f'Class (id: {id_old}) not in mapping')

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

def delete_class(dir: str, bad_id: int):
    """
    Delete a class from data using YAML file.
    """

    for filename in os.listdir(dir):

        if not filename.endswith(".txt"):
            continue

        path = os.path.join(dir, filename)
        lines = []

        with open(path, "r") as f:
            for line in f:

                parts = line.strip().split()
                id = int(parts[0])

                if id > bad_id:
                    parts[0] = id - 1
                    lines.append(" ".join(parts))
                elif id < bad_id:
                    lines.append(" ".join(parts))
                
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")


for id in bad_ids_1:
    delete_class(file_path, id)

# you need to update both datasets for bad, thankfully there are now consistent with one another. 
# one you do this, you can retrain but with correct junction annotations, which you can just go through
# and do manually ??? --> Just delete the bad ones i'll see how long this actually takes. 
# about 5k images. rf might also have model that helps remove that. 
