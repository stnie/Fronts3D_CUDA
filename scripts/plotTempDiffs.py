import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import sys

    

# where the files are located
data_fold = sys.argv[1]
# which fronts to process
ftypes = sys.argv[2]

out_path = sys.argv[3]
# filter corresponding files
files = [ x for x in os.listdir(data_fold) if ("2016" in x and "seps" in x)]

# get all files
allTypes = ftypes.split(",")

# map containing data for potential filtering of invalids.
alldata = {x: np.zeros((16,0), dtype=np.int32) for x in allTypes}
# map containing all split locations
allSeps = {x: np.zeros((16,0), dtype=np.int32) for x in allTypes}
# map containing all directions
allDirs = {x: np.zeros(0, dtype=np.int32) for x in allTypes}

for key,val in alldata.items():
    # only use a subsample
    files_type = sorted([ x for x in files if key in x])[::]
    test_factor = 1
    for idx,f in enumerate(files_type):
        print("\r ({} / {}) {}".format(idx, len(files_type),f),end="")
        fdata = f.replace("seps","diffs")
        data = np.fromfile(os.path.join(data_fold,f), dtype=np.float32).reshape(16,-1)
        
        # concatenate the value to the global map and filter
        val = np.concatenate((val, data), axis=-1)

        # get the separation file
        fsep = f

        # separation points
        sepdata = np.fromfile(os.path.join(data_fold,fsep), dtype=np.int32).reshape(16,-1)
        fdir = f.replace("seps", "dirs")
        # separation points
        
        dirdata = np.fromfile(os.path.join(data_fold,fdir), dtype=np.float32)
        allDirs[key] = np.concatenate((allDirs[key], dirdata))
        
        # filter valid seperation points
        allSeps[key] = np.concatenate((allSeps[key], sepdata), axis=-1)


    # filter data points 
    validSamps = (val > - 1000) * (val < 1000)

    orientations = allDirs[key]
    xpart,ypart = np.cos(orientations), np.sin(orientations)
    
    if("north_images" in out_path):
        validSamps *= (ypart  > np.sqrt(0.5))
    if("south_images" in out_path):
        validSamps *= (ypart  < -np.sqrt(0.5))
        

    allSeps[key][~validSamps] = 0

display_names = {
    "warm":"warm",
    "cold":"cold",
    "occ_":"occluded",
    "occ2":"flipped occluded",
    "stnry": "stationary"}

if("north_images" in out_path or "south_images" in out_path):
    allSeps["occ2"] = np.concatenate((allSeps["occ2"], allSeps["occ_"]), axis=-1)
    allDirs["occ2"] = np.concatenate((allDirs["occ2"], allDirs["occ_"]), axis=-1)
    if("north_images" in out_path):
        display_names = {
        "warm":"warm",
        "cold":"cold",
        "occ_":"occluded",
        "occ2":"northward occluded",
        "stnry": "stationary"}
    if("south_images" in out_path):
        display_names = {
        "warm":"warm",
        "cold":"cold",
        "occ_":"occluded",
        "occ2":"southward occluded",
        "stnry": "stationary"}


# create inclination plots
myImages = {}
for key,val in allSeps.items():
    image = np.zeros((16,80,3))
    place = 40
    for h in range(image.shape[0]-1,-1,-1):
        alldiffs = val[h]-val[-1]
        valids = abs(alldiffs) < 40
        valids *= val[h]!= 0
        valids *= val[-1] != 0
        places, counts = np.unique(alldiffs[valids], return_counts=True)
        places += 40
        # draw into image
        normFactor = valids.sum()
        image[h, places,0] += counts/normFactor
        image[h, places,1] += counts/normFactor
        image[h, places,2] += counts/normFactor
            
    myImages[key] = image
    myImages[key][:,0]=0

# values for axes 
levels=[500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000]
step_size = 20

for key,val in allSeps.items():   
    DirFig = go.Figure()
    DirFig.add_trace(go.Image(z = (myImages[key]*255*10).astype(np.float32),name = key))
    DirFig.update_layout(
        title = "{} fronts".format(display_names[key]),
        title_x = 0.15,
        title_y = 0.3,
        height=150,
        title_font=dict(color="white"),
        margin = dict(l=0,r=5,t=0,b=0),
        xaxis_title = 'km',
        yaxis_title = 'pressure (hPa)',
        yaxis_tickvals = np.arange(0,len(levels),5),
        yaxis_ticktext = levels[::5],
        xaxis_tickvals = np.arange(0,myImages[key].shape[1],10),
        xaxis_ticktext = step_size*np.arange(-myImages[key].shape[1]/2, myImages[key].shape[1]/2, 10))
    DirFig.update_yaxes(automargin=True)
    DirFig.update_xaxes(automargin=True)
    DirFig.write_image(os.path.join(out_path,"mean_direction_{}.svg".format(key)))
