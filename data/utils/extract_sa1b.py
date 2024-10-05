import os
import subprocess
from multiprocessing import Pool

def extract(i):
    file = f"sa_{i:06}.tar"
    dir = os.path.join(parent, f"sa_{i:06}")
    if os.path.exists(file):
        print(f"Extracting {file} to {dir}")
        os.makedirs(dir, exist_ok=True)
        subprocess.call(["tar", "--skip-old-files", "-xf", file, "-C", dir])
        print(f"Extracted {file}")
    else:
        print(f"File {file} does not exist")

if __name__ == "__main__":
    os.chdir("/cpfs/shared/research-llm/liujianfeng/08_subobject/data/OpenDataLab___SA-1B/raw")
    parent = "/cpfs/shared/research-llm/liujianfeng/08_subobject/data/sa1b"
    os.makedirs(parent, exist_ok=True)
    with Pool() as p:
        p.map(extract, range(0,1000))