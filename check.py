import os

print("TRAIN FIGHT:")
print(os.listdir("train/fight")[:5])

print("\nTRAIN NONFIGHT:")
print(os.listdir("train/nonfight")[:5])

print("\nTEST FIGHT:")
print(os.listdir("test/fight")[:5])

print("\nTEST NONFIGHT:")
print(os.listdir("test/nonfight")[:5])