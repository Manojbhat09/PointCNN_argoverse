import os
import time
import shutil
import glob
import sys
import pandas as pd
from tqdm import tqdm
from subprocess import Popen, PIPE
import argparse

parser = argparse.ArgumentParser(description="Data conversion from rosbag to bin files")
parser.add_argument("--TAG", "-T", required=True, default="bosch1", help="`TAG` which is the name of the dataset you want to save. If `TAG = \"bosch1\"` then the saved dataset folder will be `8_bosch1_0000_sync` where the number `8` depends on the available number of datasets in the `test_dataset` folder. If there are 7 datasets, the next is named `8_XXX_000_sync`. This is to follow the `KITTI` and `LATTE` dataset standard. The `.bin` files after conversion will be saved at `X_YYY_000_sync/bin_files/`.")
parser.add_argument("--CREATE_DATASET", "-CD",  action="store_true", help=" `CREATE_DATASET` is a flag which should be TRUE if a new dataset needs to be created from the rosbag data. Otherwise the data will be saved inside a folder in the same folder where the bag files are located i.e `<Folder_Name>/app/rosbags/X`")
parser.add_argument("--CONVERT_FILES", "-CF", action="store_true", help="`CONVERT_FILES` is a flag which should be TRUE if a need is to convert the intermediate `.pcd` data into `.bin` data. Otherwise the `.pcd` files will not be converted.")
parser.print_help()
args = parser.parse_args()

TAG = args.TAG #"bosch1" # The name of the dataset
CREATE_DATASET = args.CREATE_DATASET #1
CONVERT_FILES = args.CONVERT_FILES #1

def create_dataset(dataset_root_dir, additional_count):
    # Exploring present dataset/folder names
    dirnames = os.listdir(dataset_root_dir+"/")

    current_numbers = list()
    for each in dirnames:
        list_name = each.split("_")
        try:
            current_numbers.append(int(list_name[0]))
        except Exception as e:
            print("Not counting dataset name :", each)
            continue
    last_number = sorted(current_numbers)[-1]
    new_number = last_number+1+ additional_count

    # Making a new dataset given the tag name
    folder_name = str(new_number) +"_{}_0000_sync".format(TAG)
    dataset_path = os.path.join(dataset_root_dir, folder_name)
    print("#"*100 + "\n" + "Created new dataset named: {} \nDataset path: {}".format(folder_name, dataset_path) + "\n" + "#"*100)
    os.makedirs(os.path.join(dataset_path, "bin_data"))
    os.makedirs(os.path.join(dataset_path, "ground_removed"))
    os.makedirs(os.path.join(dataset_path, "image"))
    os.makedirs(os.path.join(dataset_path, "oxts"))
    return dataset_path

def pcd_to_bin(pcd_files_paths, intensity = True):
    # Importing only if conversion required
    from pyntcloud import PyntCloud
    import numpy as np
    
    cols = 4 if intensity else 3
    
    try: 
        current_path = os.path.dirname(pcd_files_paths[0])
    except Exception as e:
        print(e)
        print("No pcds found in current folder, issue with bagfile or ros.")
    count = 0
    
    print("Replacing pcd files with bin files ...")
    time.sleep(3)
    
    new_paths = list()
    for pcd_path in tqdm(np.sort(pcd_files_paths)): 
        pcd_file_name = pcd_path[len(current_path)+1:-4]
        pc = PyntCloud.from_file(os.path.join(pcd_path))
        points_df = pc.points
        xyzi2=points_df.to_numpy()[ : , : cols]
        xyzi2 = xyzi2.reshape((-1, cols))
        xyzi2 = xyzi2[np.logical_not(np.isnan(xyzi2[:,0]))] # removing all nan points
        xyzi2 = xyzi2[np.logical_not(np.isnan(xyzi2[:,1]))]
        xyzi2 = xyzi2[np.logical_not(np.isnan(xyzi2[:,2]))]
        if intensity:
            xyzi2 = xyzi2[np.logical_not(np.isnan(xyzi2[:,3]))]
        np.save(current_path+"/{}.bin".format(pcd_file_name), xyzi2)
        
        src = current_path+"/{}.bin.npy".format(pcd_file_name)
        dst = current_path+"/{:06d}.bin".format(int(count)) 
        if os.path.exists(src):
            shutil.move(src, dst)
            
        os.remove(pcd_path)
        new_paths.append(dst)
        count+=1
        
    print("Replaced all pcd files ...")
    return np.array(new_paths)

def bag_to_pcd(bag_name):
    try:
        process = Popen(["sh", "./ros_process.sh", bag_name], stdout = PIPE, stderr = PIPE)

        while True:
            nextline = process.stdout.readline()
            if str(nextline) == '' or process.poll() is not None or str(nextline) in 'killed':
                break
            sys.stdout.write(str(nextline))
            sys.stdout.flush()
        skip=0

    except Exception as e:
        print(e)
        print("Check if roscore is running, or there might be an issue with the bag file. Use \"rostopic list\" to see topics of interest")
        skip=1
    return skip

if __name__ == "__main__":
    
    start = time.time()
    # convert bag to bin into dataset
    current_path = os.path.dirname(os.path.realpath(__file__))
    bag_list = glob.glob(current_path+"/*.bag")
    folder_count = 0
    for each_bag in bag_list:
        bag_name = each_bag[len(current_path)+1:]
        skip = bag_to_pcd(bag_name) # bag_name = "2019-09-15-15-40-36.bag" (Example)
        if skip:
            print("Skipping bag ", bag_name)
            continue

        if not CREATE_DATASET:
            folder_path = os.path.join(current_path, str(folder_count))
            os.makedirs(folder_path, exist_ok=True)
            # folder_path = os.path.join(current_path, "..", "pcdfiles") (Example) 
        else:
            dataset_root = os.path.join(current_path, "..", "test_dataset")
            dataset_path = create_dataset(dataset_root, folder_count)
            folder_path = os.path.join(dataset_path, "bin_data") + "/"

        files_path = glob.glob(current_path + "/*.pcd")
        if CONVERT_FILES:
            files_path = pcd_to_bin(files_path, intensity=True)

        # Move all the files into dataset
        src_files = files_path
        dest = folder_path
        for src in src_files:
            shutil.move(src, dest+"/")
            
        print("Files from bag file {} moved to folder {}".format(bag_name, dest))
        print("Time taken for dataset make: {} s".format(time.time()-start))
        folder_count += 1
        start = time.time()
        