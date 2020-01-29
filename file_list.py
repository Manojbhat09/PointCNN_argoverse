import os
import numpy as np
# SIZE = 15000

# def train_list(path):
#     f = open(path,'w')
#     count = 0
#     hashes = os.listdir('/home/kartik/DL_model/PRCNN_Orig/PointRCNN/data/Argo/train/')
    
#     for folder in hashes:
#         filenames = os.listdir('/home/kartik/DL_model/PRCNN_Orig/PointRCNN/data/Argo/train/' + folder + '/lidar/')
#         bool_mask = np.random.choice(a = [True,False], size = (len(filenames),), p=[0.8,0.2])
        
#         for pid,name in enumerate(filenames):
#             if bool_mask[pid] and count<SIZE:
#                 if (name.split('.')[-1] == 'ply'):
#                     f.write(folder + ',' + str(name).split('.')[0].split('_')[-1])
#                     f.write('\n')
#                     count += 1
#     print(count)
#     f.close()

# if __name__ == '__main__':
#     train_list('./data/KITTI/ImageSets/train.txt')

# SIZE = 15000
# def train_list(path):
#     f = open(path,'w')
#     count = 0
#     hashes = os.listdir('/home/kartik/DL_model/argoverse-api/data/test/')
    
#     for folder in hashes:
#         filenames = os.listdir('/home/kartik/DL_model/argoverse-api/data/test/' + folder + '/lidar/')
#         bool_mask = np.random.choice(a = [True,False], size = (len(filenames),), p=[1,0])
        
#         for pid,name in enumerate(filenames):
#             if bool_mask[pid] and count<SIZE:
#                 if (name.split('.')[-1] == 'ply'):
#                     f.write(folder + ',' + str(name).split('.')[0].split('_')[-1])
#                     f.write('\n')
#                     count += 1
#     print(count)
#     f.close()

# if __name__ == '__main__':
#     train_list('./data/KITTI/ImageSets/test.txt')


# SIZE = 15000
# def train_list(path):
#     f = open(path,'w')
#     count = 0
#     hashes = os.listdir('/home/kartik/DL_model/PointRCNN_Manoj/data/Argo/val/')
    
#     for folder in hashes:
#         filenames = os.listdir('/home/kartik/DL_model/PointRCNN_Manoj/data/Argo/val/' + folder + '/lidar/')
#         bool_mask = np.random.choice(a = [True,False], size = (len(filenames),), p=[1,0])
        
#         for pid,name in enumerate(filenames):
#             if bool_mask[pid] and count<SIZE:
#                 if (name.split('.')[-1] == 'ply'):
#                     f.write(folder + ',' + str(name).split('.')[0].split('_')[-1])
#                     f.write('\n')
#                     count += 1
#     print(count)
#     f.close()

# if __name__ == '__main__':
#     train_list('./data/KITTI/ImageSets/val.txt')




# SIZE = 150
# cwd = os.getcwd()
# def train_list(path):
#     f = open(path,'w')
#     count = 0
#     hashes = os.listdir(os.path.join(cwd, 'data', "Argo", "train"))
    
#     for folder in hashes:
#         filenames = os.listdir(os.path.join(cwd, 'data', "Argo", "train", folder, "lidar") + '/')
#         bool_mask = np.random.choice(a = [True,False], size = (len(filenames),), p=[1,0])
        
#         for pid,name in enumerate(filenames):
#             if bool_mask[pid] and count<SIZE:
#                 if (name.split('.')[-1] == 'ply'):
#                     f.write(folder + ',' + str(name).split('.')[0].split('_')[-1])
#                     f.write('\n')
#                     count += 1
#     print(count)
#     f.close()

# if __name__ == '__main__':
#     train_list('./data/KITTI/ImageSets/train.txt')


# SIZE = 150
# cwd = os.getcwd()
# def train_list(path):
#     f = open(path,'w')
#     count = 0
#     hashes = os.listdir(os.path.join(cwd, 'data', "Argo", "train"))
    
#     for folder in hashes:
#         filenames = os.listdir(os.path.join(cwd, 'data', "Argo", "train", folder, "lidar") + '/')
#         bool_mask = np.random.choice(a = [True,False], size = (len(filenames),), p=[1,0])
        
#         for pid,name in enumerate(filenames):
#             if bool_mask[pid] and count<SIZE:
#                 if (name.split('.')[-1] == 'ply'):
#                     f.write(folder + ',' + str(name).split('.')[0].split('_')[-1])
#                     f.write('\n')
#                     count += 1
#     print(count)
#     f.close()

# if __name__ == '__main__':
#     train_list('./data/Argo/train.txt')


# H5
# SIZE = 20
# cwd = os.getcwd()
# def train_list(path):
#     f = open(path,'w')
#     count = 0
    
#     filenames = os.listdir(os.path.join(cwd, 'data', "Argo_h5", "train") + '/')
#     bool_mask = np.random.choice(a = [True,False], size = (len(filenames),), p=[1,0])
#     print("files are ", len(filenames))
#     for pid,name in enumerate(sorted(filenames)):
#         if bool_mask[pid] and count<SIZE:
#             if (name.split('.')[-1] == 'h5'):
#                 f.write(name)
#                 f.write('\n')
#                 count += 1
#     print(count)
#     f.close()

# if __name__ == '__main__':
#     train_list('./data/Argo_h5/train/train.txt')
    
SIZE = 1
cwd = os.getcwd()
def train_list(path):
    f = open(path,'w')
    count = 0
    
    filenames = os.listdir(os.path.join(cwd, 'data', "Argo_h5_GR_scaled_depth", "train") + '/')
    bool_mask = np.random.choice(a = [True,False], size = (len(filenames),), p=[1,0])
    print("files are ", len(filenames))
    for pid,name in enumerate(sorted(filenames)):
        if bool_mask[pid] and count<SIZE:
            if (name.split('.')[-1] == 'h5'):
                f.write(name)
                f.write('\n')
                count += 1
    print(count)
    f.close()

if __name__ == '__main__':
    train_list('./data/Argo_h5_GR_scaled_depth/train/val.txt')
    
    
    

