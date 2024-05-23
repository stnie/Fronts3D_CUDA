import numpy as np
import os
import sys


# undisturbed
compare_data_fold = sys.argv[1]

#disturbed
target_data_fold = sys.argv[2]

front_type = sys.argv[3]
mylevel = int(sys.argv[4])
if(len(sys.argv)>5):
    num_files = int(sys.argv[5])
else:
    num_files = None

files = sorted([file for file in os.listdir(compare_data_fold) if "{}_segs".format(front_type) in file])

allMedianDists = []
consistentDir = []

# fixed number (2 interpolations with 16 points each)
segsize = 32 
for fileidx,file in enumerate(files[:num_files]):
    print("file {} {} / {}".format(file, fileidx, len(files)), end= "\r")
    fsep = file
    segments_compare = np.fromfile(os.path.join(compare_data_fold, fsep), dtype=np.int64).reshape(-1)*segsize
    segments_target = np.fromfile(os.path.join(target_data_fold, fsep), dtype=np.int64).reshape(-1)*segsize
    
    fsepdir = fsep.replace("_segs", "_dirs")
    dirs_compare = np.fromfile(os.path.join(compare_data_fold, fsepdir), dtype=np.float32).reshape(-1)
    dirs_target = np.fromfile(os.path.join(target_data_fold, fsepdir), dtype=np.float32).reshape(-1)
    
    fsep = fsep.replace("_segs", "")
    locs_compare = np.fromfile(os.path.join(compare_data_fold, fsep), dtype=np.float32).reshape(16,-1,2)
    locs_target = np.fromfile(os.path.join(target_data_fold, fsep), dtype=np.float32).reshape(16,-1,2)

    
    for segidx in range(segments_compare.shape[0]-1):
        assert(segments_compare[segidx] == segments_target[segidx])
        front_compare = locs_compare[mylevel, segments_compare[segidx]:segments_compare[(segidx+1)]]
        front_target = locs_target[mylevel, segments_target[segidx]:segments_target[(segidx+1)]]
        num_segments = (segments_compare[segidx+1]-segments_compare[segidx])//segsize
        # ignore very short fronts
        if(num_segments <= 2): continue
        # get directions to check consistent orientation
        compare_dirs = dirs_compare[segments_compare[segidx]:segments_compare[(segidx+1)]]
        target_dirs = dirs_target[segments_compare[segidx]:segments_compare[(segidx+1)]]
        direction_are_same = np.cos(compare_dirs-target_dirs) > 0.9 
        consistentDir.append(direction_are_same.sum()  == direction_are_same.shape[0])
        if not (direction_are_same.sum() == 0 or direction_are_same.sum() == direction_are_same.shape[0]//2 or direction_are_same.sum() == direction_are_same.shape[0]):
            print("WARNING: This should not occur, exit")
            exit(1)
    
        # create 2d matrix of all to all distances
        pointDiffs = front_compare[None]-front_target[:,None]
        # rows are target, columns are compare
        pointDistances = np.linalg.norm(pointDiffs,axis=-1)
        # reduce to get the minimal distance of each point in compare to the next point in target
        minDists = np.min(pointDistances, axis=0)
    
        distances = minDists
        medianMin = np.median(distances)
        allMedianDists.append(medianMin)
        


allMedianDists = np.array(allMedianDists)

consistentDir = np.array(consistentDir)

print()
print("Dir flips: {}".format(consistentDir.shape[0]-consistentDir.sum()))

print("Median min dist: {}".format(allMedianDists.mean()))
print("Median min dist: {}".format(allMedianDists[consistentDir].mean()))
    


