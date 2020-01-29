import numpy as np
from tqdm import tqdm
import importlib
import tensorflow as tf
import glob
import argparse
import sys


def check_presence(bin_name, DATA_DIR, drivename,fname):
    if(not path.exists(bin_name)):
        print("File not present! Check location: ", bin_name)
        bin_name = os.path.join(DATA_DIR, drivename, "bin_data", fname) + ".npy"
        if(not path.exists(bin_name)):
            print("File not present! Check location: ", bin_name)
            bin_name = os.path.join(DATA_DIR, drivename, "bin_data", fname) + ".bin.npy"
        else:
            print("File present. Location: ", bin_name)
    else:
        print("File present. Location: ", bin_name)
    return bin_name


def add_intensity(new_points):
    new_points = new_points.reshape((-1, 3))
    new_points = np.hstack((new_points[:int(len(new_points)/3),:],np.ones((int(len(new_points)/3) , 1))))
    return new_points



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dirpath',
        '-dp',
        type = str,
        default = '/home/manojbhat09_photo/PointCNN',
        help = "directory path to process "
    )
    parser.add_argument(
        '--drivename',
        '-dn',
        type = str,
        default = '0',
        help = "Drive name  "
    )
    parser.add_argument(
        '--filename',
        '-fn',
        type = str,
        default = '000000',
        help = "File name "
    )
    parser.add_argument(
        '--groundremove',
        '-g',
        type = int,
        default = 0,
        help = "ground_removed or not? 1/0"
    )

    FLAGS, unparsed = parser.parse_known_args()
    drivename = FLAGS.drivename
    fname = FLAGS.filename
    ROOT_DIR = os.path.dirname(FLAGS.dirpath)
    ground_removed = FLAGS.groundremove

    # Directory to get binary
    PARENT_DIR = ROOT_DIR 
    DATA_DIR = os.path.join(PARENT_DIR, "/test_dataset")
    print(DATA_DIR)    

    # check ground removed
    if(ground_removed == 1):        
        bin_name = os.path.join(DATA_DIR, drivename, "ground_removed", fname) + ".bin"
        bin_name = check_presence(bin_name,DATA_DIR, drivename,fname)
    else:
        bin_name = os.path.join(DATA_DIR, drivename, "bin_data", fname) + ".bin"
        bin_name = check_presence(bin_name,DATA_DIR, drivename,fname)

    print(ground_removed, bin_name)


    # Determine appropriate loading of the file.
    new_points = np.fromfile(os.path.join(bin_name), dtype=np.float32)
    if new_points[0] >= 5e+02:
        new_points = np.load(os.path.join(bin_name))
    else:
        pass 

    # if last column not present add, last column
    if len(new_points.shape) == 1:
        if len(new_points) % 4 != 0:
            if (len(new_points.flatten()) %3 == 0):
                new_points = add_intensity(new_points)
            else:
                print("Dimentions not correct, check data. matrix is not Nx3 nor Nx4")
        else:
            new_points = new_points.reshape((-1, 4))
    else:
        if(new_points.shape[-1] == 3):
            print("The shape of the loaded file array is NX3 without intensity shape: ", new_points.shape)
            new_points = add_intensity(new_points)
        if(new_points.shape[-1] == 4):
            print("The shape of the loaded file array is NX4 with intensity shape: ", new_points.shape)
            new_points = new_points.reshape((-1, 4))
        else:
            print("The shape[1] of array is more than 4, exiting ...")
            exit()


    # remove nan noise
    new_points = new_points[np.logical_not(np.isnan(new_points[:,0]))]
    new_points = new_points[np.logical_not(np.isnan(new_points[:,1]))]
    new_points = new_points[np.logical_not(np.isnan(new_points[:,2]))]
    new_points = new_points[np.logical_not(np.isnan(new_points[:,3]))] 

    # if last column not normalized, normalize
    int_points = new_points[:,3]
    normal_int_points = (int_points - np.min(int_points))/(np.max(int_points) - np.min(int_points))
    # standard_int_points = (int_points - np.mean(int_points))/(np.sqrt(np.var(int_points)))
    new_points[:,3] = normal_int_points
    new_points.tofile(os.path.join(bin_name+"_2"))

    pca = PCA(n_components = 3)
    pca.fit(np.vstack([X[:,2], X[:,1], X[:,0]]).T)  

    axis_sort = np.argsort(pca.mean_)
    axis_dict = {2:'Z', 1:'X', 0:'Y'}
    axis_dict[axis_sort[0]]
    print("Axis columns: 0:{}, 1:{}, 2:{}".format(axis_dict[axis_sort[0]], axis_dict[axis_sort[1]], axis_dict[axis_sort[2]]))
