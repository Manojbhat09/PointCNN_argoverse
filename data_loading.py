import os
import sys
import math
import h5py
import argparse
import importlib
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
from tqdm import tqdm

import argoverse 
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from pyntcloud import PyntCloud
from argoverse.map_representation.map_api import ArgoverseMap

import data_utils
import object3d

import json
from functools import reduce
import math 
from multiprocessing import current_process, Pool
import multiprocessing

import pdb 

# Sane model
# object_types = {'Pedestrian':1, 'Car': 2, 'Cyclist': 3, 'Truck': 4,     'Person_sitting' : 5, 'Motorbike' : 6, 'Trailer' : 7, 'Bus' : 8, 'Railed' : 9, 'Airplane' : 10, 'Boat' : 11, 'Animal' :12, 'DontCare' : 13, 'Misc' : 14, 'Van' : 15, 'Tram' : 16, 'Utility' : 17}

# actual
# object_dict = {
#     'PEDESTRIAN': 1,
#     'BICYCLE' : 6,
#     'BICYCLIST' : 3,
#     'MOTORCYCLE':6,
#     'MOTORCYCLIST':5,
#     'MOPED':6,
#     'STROLLER':15,
    
#     'VEHICLE':2,
#     'TRAILER':7,
#     'LARGE_VEHICLE':4,
#     'BUS': 8,
#     'OTHER_MOVER':15,
    
#     'EMERGENCY_VEHICLE':14,
#     'ANIMAL':12,
#     'ON_ROAD_OBSTACLE':13 
# }

# Training
object_dict = {
    'PEDESTRIAN': 1,
    'BICYCLE' : 3,
    'BICYCLIST' : 3,
    'MOTORCYCLE':3,
    'MOTORCYCLIST':3,
    
    'VEHICLE':2,
#     'TRAILER':4,
#     'LARGE_VEHICLE':4,
#     'BUS': 4,
    
    'ON_ROAD_OBSTACLE':0 
}


# olda data
# object_dict = {
#     'PEDESTRIAN': 1,
#     'BICYCLE' : 2,
#     'BICYCLIST' : 3,
#     'MOTORCYCLE':4,
#     'MOTORCYCLIST':5,
#     'MOPED':6,
#     'STROLLER':7,
    
#     'VEHICLE':8,
#     'TRAILER':9,
#     'LARGE_VEHICLE':10,
#     'BUS': 11,
#     'OTHER_MOVER':12,
    
#     'EMERGENCY_VEHICLE':13,
#     'ANIMAL':14,
#     'ON_ROAD_OBSTACLE':15 
# }

argo_to_kitti = np.array([[6.927964e-03, -9.999722e-01, -2.757829e-03],
                               [-1.162982e-03, 2.749836e-03, -9.999955e-01],
                               [9.999753e-01, 6.931141e-03, -1.143899e-03]])  

parser = argparse.ArgumentParser()
parser.add_argument('--folder', '-f', help='Path to data folder')
parser.add_argument('--save_folder', '-sf', help='Path to save folder')
parser.add_argument('--max_point_num', '-m', help='Max point number of each sample', type=int, default=8192)
parser.add_argument('--block_size', '-b', help='Block size', type=float, default=5.0)
parser.add_argument('--grid_size', '-g', help='Grid size', type=float, default=0.1)
# parser.add_argument('--save_ply', '-s', help='Convert .pts to .ply', action='store_true')
parser.add_argument('--ground_removed', '-gr', help='Add ground removal?', action='store_true')
parser.add_argument('--bbox_extend', '-be', help='Ratio (x = 20/100) of scaling bboxs', type=float, default=0.0)
parser.add_argument('--h5_batch_size', '-bs', help="Batch size of h5 data (in each file). Default = 250", type=int, default= 250)

args = parser.parse_args()
print(args)
    


def get_objects_from_label(label_file):
    # Opens a label file, and passes the object to Object3d object, Read the json GT labels
    
    f = open(label_file)
    label_data = json.load(f) 
    objects = [object3d.Object3d(data) for data in label_data]
    return objects

def filter_pointcloud(bbox, pointcloud):
    theta = bbox.ry #["angle"]
    transformed_pointcloud = homogeneous_transformation(pointcloud[:, :3], bbox.pos[:3], -theta)#["center"]
    if bbox.l > bbox.w:
        length = bbox.l
        width = bbox.w
    else:
        length = bbox.w
        width = bbox.l
            
#     indices = np.intersect1d(np.where(np.abs(transformed_pointcloud[:,0]) <= width/2)[0], 
#                              np.where(np.abs(transformed_pointcloud[:,1]) <= length/2)[0])
    
    indices = reduce(np.intersect1d, (np.where(np.abs(transformed_pointcloud[:,0]) <= width/2)[0], 
                            np.where(np.abs(transformed_pointcloud[:,1]) <= length/2)[0],
                            np.where(np.abs(transformed_pointcloud[:,2]) <= bbox.h/2)[0]))
    return indices, pointcloud[indices,:]

def homogeneous_transformation(points, translation, theta):
    return (points[:, :3] - translation).dot(rotation_matrix_3D(theta).T)

def rotation_matrix_3D(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], 
                     [np.sin(theta), np.cos(theta), 0],
                     [0,0,1]])


def data_preprocessing(data_input ,labels,  max_point_num):
    batch_size = 2048
    block_size = 1000
    grid_size = 0.25
        
    assert len(labels) == data_input.shape[0]
#     data = np.zeros((batch_size, max_point_num, 4))
    data = np.zeros((batch_size, max_point_num, 3))
    data_num = np.zeros((batch_size), dtype=np.int32)
    label = np.zeros((batch_size), dtype=np.int32)
    label_seg = np.zeros((batch_size, max_point_num), dtype=np.int32)
    indices_split_to_full = np.zeros((batch_size, max_point_num), dtype=np.int32)
    
    xyzi = data_input[:, :4]
    
    
    indices_for_prediction = np.arange(xyzi.shape[0]) #(xyzi[:,0] >= -5 ).nonzero()[0]
    #print("indices_for_prediction", indices_for_prediction)
    # Filter point only in front on of ego-sensors
    xyzif =xyzi #= xyzi[xyzi[:,0] >= -5 ] 
    
    all_label_pred = np.zeros((xyzi.shape[0]),dtype=int)
    label_length = xyzif.shape[0]
    xyz =xyzif[:,0:3]
    
            
    xyz_min = np.amin(xyz, axis=0, keepdims=True)
    xyz_max = np.amax(xyz, axis=0, keepdims=True)
    block_size = (2 * (xyz_max[0, 0] - xyz_min[0, 0]), 2 * (xyz_max[0, 1] - xyz_min[0, 1]) ,  2 * (xyz_max[0, -1] - xyz_min[0, -1]))
    
    xyz_blocks = np.floor((xyz - xyz_min) / block_size).astype(np.int)

    #print('{}-Collecting points belong to each block...'.format(datetime.now(), xyzrcof.shape[0]))
    blocks, point_block_indices, block_point_counts = np.unique(xyz_blocks, return_inverse=True,
                                                                return_counts=True, axis=0)
    block_point_indices = np.split(np.argsort(point_block_indices), np.cumsum(block_point_counts[:-1]))
    #print('{}-{} is split into {} blocks.'.format(datetime.now(), dataset, blocks.shape[0]))

    block_to_block_idx_map = dict()
    for block_idx in range(blocks.shape[0]):
        block = (blocks[block_idx][0], blocks[block_idx][1])
        block_to_block_idx_map[(block[0], block[1])] = block_idx

    # merge small blocks into one of their big neighbors
    block_point_count_threshold = max_point_num / 3
    #print("block_point_count_threshold",block_point_count_threshold)
    nbr_block_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, 1), (1, 1), (1, -1), (-1, -1)]
    block_merge_count = 0
    for block_idx in range(blocks.shape[0]):
        if block_point_counts[block_idx] >= block_point_count_threshold:
            #print(block_idx, block_point_counts[block_idx])

            continue


        block = (blocks[block_idx][0], blocks[block_idx][1])
        for x, y in nbr_block_offsets:
            nbr_block = (block[0] + x, block[1] + y)
            if nbr_block not in block_to_block_idx_map:
                continue

            nbr_block_idx = block_to_block_idx_map[nbr_block]
            if block_point_counts[nbr_block_idx] < block_point_count_threshold:
                continue


            #print(block_idx, nbr_block_idx, block_point_counts[nbr_block_idx])

            block_point_indices[nbr_block_idx] = np.concatenate(
                [block_point_indices[nbr_block_idx], block_point_indices[block_idx]], axis=-1)
            block_point_indices[block_idx] = np.array([], dtype=np.int)
            block_merge_count = block_merge_count + 1
            break
    #print('{}-{} of {} blocks are merged.'.format(datetime.now(), block_merge_count, blocks.shape[0]))

    idx_last_non_empty_block = 0
    for block_idx in reversed(range(blocks.shape[0])):
        if block_point_indices[block_idx].shape[0] != 0:
            idx_last_non_empty_block = block_idx
            break

    # uniformly sample each block
    for block_idx in range(idx_last_non_empty_block + 1):
        point_indices = block_point_indices[block_idx]
        if point_indices.shape[0] == 0:
            continue

        #print(block_idx, point_indices.shape)
        block_points = xyz[point_indices]
        block_min = np.amin(block_points, axis=0, keepdims=True)
        xyz_grids = np.floor((block_points - block_min) / grid_size).astype(np.int)
        grids, point_grid_indices, grid_point_counts = np.unique(xyz_grids, return_inverse=True,
                                                                 return_counts=True, axis=0)
        grid_point_indices = np.split(np.argsort(point_grid_indices), np.cumsum(grid_point_counts[:-1]))
        grid_point_count_avg = int(np.average(grid_point_counts))
        point_indices_repeated = []
        for grid_idx in range(grids.shape[0]):
            point_indices_in_block = grid_point_indices[grid_idx]
            repeat_num = math.ceil(grid_point_count_avg / point_indices_in_block.shape[0])
            if repeat_num > 1:
                point_indices_in_block = np.repeat(point_indices_in_block, repeat_num)
                np.random.shuffle(point_indices_in_block)
                point_indices_in_block = point_indices_in_block[:grid_point_count_avg]
            point_indices_repeated.extend(list(point_indices[point_indices_in_block]))
        block_point_indices[block_idx] = np.array(point_indices_repeated)
        block_point_counts[block_idx] = len(point_indices_repeated)

    idx = 0
    for block_idx in range(idx_last_non_empty_block + 1):
        point_indices = block_point_indices[block_idx]
        if point_indices.shape[0] == 0:
            continue

        block_point_num = point_indices.shape[0]
        block_split_num = int(math.ceil(block_point_num * 1.0 / max_point_num))
        point_num_avg = int(math.ceil(block_point_num * 1.0 / block_split_num))
        point_nums = [point_num_avg] * block_split_num
        point_nums[-1] = block_point_num - (point_num_avg * (block_split_num - 1))
        starts = [0] + list(np.cumsum(point_nums))

        np.random.shuffle(point_indices)
        block_points = xyz[point_indices]


        block_min = np.amin(block_points, axis=0, keepdims=True)
        block_max = np.amax(block_points, axis=0, keepdims=True)
        #block_center = (block_min + block_max) / 2
        #block_center[0][-1] = block_min[0][-1]
        #block_points = block_points - block_center  # align to block bottom center
        x, y, z = np.split(block_points, (1, 2), axis=-1)

#         block_xzyrgbi = np.concatenate([x, z, y, i[point_indices]], axis=-1)
        block_xzyrgbi = np.concatenate([x, z, y], axis=-1)
        block_labels = labels[point_indices]

        for block_split_idx in range(block_split_num):
            start = starts[block_split_idx]
            point_num = point_nums[block_split_idx]
            #print(block_split_num, block_split_idx, point_num )



            end = start + point_num
            idx_in_batch = idx % batch_size
            data[idx_in_batch, 0:point_num, ...] = block_xzyrgbi[start:end, :]
            data_num[idx_in_batch] = point_num
            
            label_seg[idx_in_batch, 0:point_num] = block_labels[start:end]
            
            indices_split_to_full[idx_in_batch, 0:point_num] = point_indices[start:end]

            #print("indices_split_to_full", idx_in_batch, point_num, indices_split_to_full)

            if  (block_idx == idx_last_non_empty_block and block_split_idx == block_split_num - 1): #Last iteration

                item_num = idx_in_batch + 1
                
            idx = idx + 1
            
    return label_length, data, data_num, label_seg, indices_split_to_full, item_num, all_label_pred, indices_for_prediction


def compute_dataset( file_paths, h5_batches, folder_choice, save_folder, extend_factor = 0.1, ground_removed = True):
    
    p = current_process()
    print('process counter:', p._identity[0]-1, " till: ", (p._identity[0]-1)*h5_batches,  'pid:', os.getpid())
    process_idx = p._identity[0]
    
    max_point_num = 8192
    batch_size = 2048

    data_all= list()
    data_num_all= list()
    label_all= list()
    label_seg_all= list()
    indices_all= list()
    index_list1= list()
    index_list2= list()
    
    id_h5 = (p._identity[0]-1)
    
    # filename_h5 = os.path.join(save_h5_root , folders[0], folders[0]+ '_%s_%d.h5' % ("full", data_idx))
    folder_name = folder_choice[len(os.path.dirname(folder_choice))+1:]
    filename_h5 = os.path.join(save_folder, folder_name, folder_name+ '_%s_%d.h5' % ("full", id_h5))
    
    if(os.path.exists(filename_h5)):
        print("skipping ", filename_h5, " <file exists>")
        return
    
#     name = [each[len(os.path.dirname(text))+1:-4] for each in file_paths]
#     pdb.set_trace()

    for idx, dataset_file in enumerate(tqdm(sorted(file_paths))):

        data_idx = (p._identity[0]-1)*h5_batches + idx
        
        # Get lidar points
        actual_index = actual_idx_list[data_idx]
        index_list1.append(actual_index[0])
        index_list2.append(actual_index[1])
        data_points = PyntCloud.from_file(dataset_file)

        if(ground_removed):
            folder = actual_index[0]
            dataset = int(actual_index[1])

            log, idx = actual_index[0], int(actual_index[1])
            lidar_file = dataset_file

            argoverse_data = data_loader.get(log)
            city_name = argoverse_data.city_name

            log_dataidx_list = log_to_count_map[log]
            log_data_idx = log_dataidx_list.tolist().index(data_idx)

            lidar_pts = argoverse_data.get_lidar(log_data_idx)
            city_to_egovehicle_se3 = argoverse_data.get_pose(log_data_idx)
            roi_area_pts = city_to_egovehicle_se3.transform_point_cloud(lidar_pts) # more to city CS
            roi_area_pts = am.remove_non_roi_points(roi_area_pts, city_name) # remove outside roi points
            roi_area_pts = am.remove_ground_surface(roi_area_pts, city_name) # remove ground  
            roi_area_pts = city_to_egovehicle_se3.inverse_transform_point_cloud(
                roi_area_pts
            )# Back to lidar cs

            x = np.array(roi_area_pts[:,0])[:, np.newaxis]
            y = np.array(roi_area_pts[:,1])[:, np.newaxis]
            z = np.array(roi_area_pts[:,2])[:, np.newaxis]
            pts_lidar = np.concatenate([x,y,z], axis = 1)
        else:
            pts_lidar = data_points.xyz
        

        # Get objects
        label_file = label_pathlist[data_idx]
        objects = get_objects_from_label(label_file)

        label_indices = list()
        labels = np.zeros(pts_lidar.shape[0])
        rgb = np.zeros(pts_lidar.shape[0])
        i = np.zeros(pts_lidar.shape[0])

        # Get labels
        for each in objects:
            
            if each.cls_type not in object_dict.keys():
                object_id = 0
            else:
                object_id = object_dict[each.cls_type]

            each.pos = np.dot(each.pos, argo_to_kitti)
            
            # 3rd modification
            each.l = each.l + extend_factor* each.l
            each.w = each.w + extend_factor* each.w
            each.h = each.h + extend_factor* each.h
            
            out = filter_pointcloud(each, np.copy(pts_lidar))

            label_indices.extend(out[0].tolist() if type(out[0]) == np.ndarray else out[0])
            labels[label_indices] = object_id
            rgb[label_indices] = len(out[0])
            i[label_indices] = object_id
            
        
        depth = np.linalg.norm(pts_lidar, 2, axis=1)
#         data_full = np.hstack((pts_lidar, rgb[:,np.newaxis]))
        data_full = np.hstack((pts_lidar, depth[:,np.newaxis]))

        label_length, data, data_num, label_seg, indices_split_to_full, item_num, all_label_pred, indices_for_prediction = data_preprocessing(data_full, labels, max_point_num)
        
        print(label_length)
        data_data =data[0:item_num, ...].astype(np.float32) 
        data_num =data_num[0:item_num, ...] 
        indices_split_to_full = indices_split_to_full[0:item_num]
        label_seg_data = label_seg[0:item_num]
        batch_num = data.shape[0]

        data_all.append(data_data)
        data_num_all.append(data_num)
        label_all.append(item_num*(data_idx+1))
        label_seg_all.append(label_seg_data)
        indices_all.append(indices_split_to_full)

    data_data = np.concatenate(data_all)
    data_num_all = np.concatenate(data_num_all)
    label_seg_all = np.concatenate(label_seg_all)
    indices_all = np.concatenate(indices_all)
    
    index_list1 = np.array(index_list1,  dtype=h5py.string_dtype(encoding='ascii'))
    index_list2 = np.array(index_list2,  dtype=h5py.string_dtype(encoding='utf-8'))

    file = h5py.File(filename_h5, 'w')
    file.create_dataset('data', data=data_data)
    file.create_dataset('data_num', data=data_num_all)
    file.create_dataset('label', data=label_all)
    file.create_dataset('label_seg', data=label_seg_all)
    file.create_dataset('indices_split_to_full', data=indices_all)
    file.create_dataset('identity1', data=index_list1)
    file.create_dataset('identity2', data=index_list2)
    file.close()

        
    print("Saved to {} ".format(data_idx), filename_h5)
    
    data_all= list()
    data_num_all= list()
    label_all= list()
    label_seg_all= list()
    indices_all= list()


    
    
    
    
    
    
    
    
    
if __name__=="__main__":   
    
    cwd = os.getcwd()

    root = args.folder if args.folder else os.path.join(cwd, "data", "Argo") + "/"
    save_h5_root = args.save_folder if args.save_folder else os.path.join(cwd, "data", "Argo_h5_GR_scaled_depth2")
    print("saving to : ", save_h5_root)
    
    folders = [os.path.join(root, folder) for folder in ['train', 'val', 'test']]
    
    folder_choice = folders[0]
    
    split = folder_choice[len(os.path.dirname(folder_choice))+1:]
    root_dir = root
    is_test = (split == 'test')
    lidar_pathlist = []
    label_pathlist = []
    actual_idx_list = []
    logidx_to_count_map= {}
    log_to_count_map= {}

    print("____________SPLIT IS : {} ______________".format(split))
    if split == 'train':
        imageset_dir = os.path.join(root_dir,split)
        splitname = lambda x: [x[len(imageset_dir+"/"):-4].split("/")[0], x[len(imageset_dir+"/"):-4].split("/")[2].split("_")[1]]
        data_loader = ArgoverseTrackingLoader(os.path.join(root_dir,split))
        log_list = data_loader.log_list
        path_count = 0
        for log_id, log in enumerate(log_list):
            lidar_lst = data_loader.get(log).lidar_list
            lidar_pathlist.extend(lidar_lst)
            label_pathlist.extend(data_loader.get(log).label_list)
            actual_idx_list.extend([splitname(each) for each in lidar_lst])
            logidx_to_count_map[log_id] = np.arange(path_count, path_count + len(lidar_lst))
            log_to_count_map[log] = np.arange(path_count, path_count + len(lidar_lst))
            path_count+=len(lidar_lst)
        assert len(lidar_pathlist) == len(label_pathlist)

    elif split == 'test':
        imageset_dir = os.path.join(root_dir,split)
        splitname = lambda x: [x[len(imageset_dir+"/"):-4].split("/")[0], x[len(imageset_dir+"/"):-4].split("/")[2].split("_")[1]]
        print("______________image set dir_________", imageset_dir)
        data_loader = ArgoverseTrackingLoader(os.path.join(root_dir,split))
        log_list = data_loader.log_list
        path_count = 0
        for log in log_list:
            lidar_pathlist.extend(data_loader.get(log).lidar_list)
            actual_idx_list.extend([splitname(each) for each in lidar_lst])
            logidx_to_count_map[log_id] = np.arange(path_count, path_count + len(lidar_lst))
            log_to_count_map[log] = np.arange(path_count, path_count + len(lidar_lst))
            path_count+=len(lidar_lst)
        print("The lidar list len is : ",len(self.lidar_pathlist))
        label_pathlist = None


    am = ArgoverseMap()
    calib_file = data_loader.calib_filename
    print("sample from lidar paths ",lidar_pathlist[0])

    num_sample = len(lidar_pathlist)
    image_idx_list = np.arange(num_sample)
    print("image list sample is: ", actual_idx_list[0])

    if len(lidar_pathlist) > len(actual_idx_list):
        print("There is length difference between lidar and actual files : ", lidar_pathlist, " ",actual_idx_list )
    else:
        print("len of list is same \n")  

    max_point_num = args.max_point_num

    batch_size = 2048

    data_all= list()
    data_num_all= list()
    label_all= list()
    label_seg_all= list()
    indices_all= list()

    id_h5 = 0
    h5_batch_size = args.h5_batch_size

    # create seperate processing pathlists
    indice = np.arange(lidar_pathlist.__len__())[::h5_batch_size]
    process_batches = int(lidar_pathlist.__len__()/h5_batch_size)
    process_path_list = [sorted(lidar_pathlist)[i:i+h5_batch_size] for i in range(process_batches)]

    print("Total process counter runs: ", process_batches/20)
    extend_factor = args.bbox_extend
    ground_removed = args.ground_removed
    
    with multiprocessing.Pool(processes=20) as pool:
        ret_list = pool.starmap(compute_dataset, [(each_list, h5_batch_size, folder_choice, save_h5_root,extend_factor ,ground_removed ) for each_list in process_path_list])

    
    
    