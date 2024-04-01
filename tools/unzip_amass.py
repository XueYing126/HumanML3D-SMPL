import os
import tarfile

def extract_tar_bz2(source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith(".tar.bz2"):
            print(f"unzip {filename}")
            with tarfile.open(os.path.join(source_dir, filename), "r:bz2") as tar:
                tar.extractall(destination_dir)

if __name__ == "__main__":
    source_directory = "amass"
    destination_directory = "amass_data"
    extract_tar_bz2(source_directory, destination_directory)
