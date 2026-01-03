import os

root = "D:\\VISHNU\\Capstone Implement\\ai-healthcare"

for path, folders, files in os.walk(root):
    for f in files:
        if f.endswith(".pkl") or f.endswith(".h5"):
            print(os.path.join(path, f))
