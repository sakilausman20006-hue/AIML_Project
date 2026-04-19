import os
import shutil
import random

base_dir = "."

train_fight = "train/fight"
train_nonfight = "train/nonfight"

test_fight = "test/fight"
test_nonfight = "test/nonfight"

os.makedirs(train_fight, exist_ok=True)
os.makedirs(train_nonfight, exist_ok=True)
os.makedirs(test_fight, exist_ok=True)
os.makedirs(test_nonfight, exist_ok=True)

for file in os.listdir(base_dir):

    if file.startswith("fi"):
        if random.random() < 0.8:
            shutil.move(file, os.path.join(train_fight, file))
        else:
            shutil.move(file, os.path.join(test_fight, file))

    elif file.startswith("no"):
        if random.random() < 0.8:
            shutil.move(file, os.path.join(train_nonfight, file))
        else:
            shutil.move(file, os.path.join(test_nonfight, file))

print("Dataset split completed")