import numpy as np
import sys
import plotly.express as px
import os
import pandas as pd
import plotly.io as pio
pio.kaleido.scope.mathjax = None

baro_folder = sys.argv[1]
front_types = sys.argv[2]
mylevel = sys.argv[3]
result_folder = sys.argv[4]

num_files_per_type = int(sys.argv[5])
filter_number = int(sys.argv[6])
outpath = sys.argv[7]

# determine if any filter should be applied ( 0 = no filter )
filter_north = filter_number>0
filter_south = filter_number<0
filtering = filter_south or filter_north


ftypes=front_types.split(",")
levels = mylevel.split(",")
for level in range(len(levels)):
    levels[level] = int(levels[level])
num_levels = len(levels)
baroResults=[[] for _ in range(num_levels)]

# 10 ahead, 10 after
width = 21

if filtering:
    myTotalSum=np.zeros((num_levels,width), dtype=np.float32)
    myTotalCounts = np.zeros((num_levels,width))

normalize_perSample = False


for front_type in ftypes:
    mySum=np.zeros((num_levels,width), dtype=np.float32)
    myCounts = np.zeros((num_levels,width))
    cs_files = sorted([file for file in os.listdir(baro_folder) if "{}_cs.bin".format(front_type) in file])
    
    print("found: {} files".format(len(cs_files)))
    if(num_files_per_type > 0):
        num_files = num_files_per_type
    else:
        num_files = len(cs_files)
    filtered = 0
    not_filtered = 0
    totalSamps = 0
    validSamps = 0
    for fidx in range(num_files):
        print("file {}/{}".format(fidx, num_files), end="\r")
        cs_file = cs_files[fidx]
        position_file = cs_file.replace("_cs","_seps")
        dir_file = cs_file.replace("_cs","_dirs")

        baros = np.fromfile(os.path.join(baro_folder, cs_file), dtype=np.float32).reshape(-1,16,256)
        
        positions = np.fromfile(os.path.join(result_folder, position_file), dtype=np.int32).reshape(16,-1)

        dirs = np.fromfile(os.path.join(result_folder, dir_file), dtype=np.float32).reshape(-1)
        
        totalSamps += baros.shape[0]

        for levelidx,level in enumerate(levels):
            for i in range(baros.shape[0]):
                if(filtering and filter_north):
                    if(np.sin(dirs[i])<np.sqrt(0.5)):
                        filtered+=1
                        continue
                    else:
                        not_filtered+=1
                elif(filtering and filter_south):
                    if(np.sin(dirs[i])>-np.sqrt(0.5)):
                        filtered+=1
                        continue
                    else:
                        not_filtered+=1
                minPos = positions[level,i]-(width//2)
                maxPos = positions[level,i]+(width//2)+1
                leftOff = min(0,max(0,-minPos))
                rightOff = width+min(0, -(maxPos-256))
                values = np.abs(baros[i,level,minPos:maxPos])
                if(normalize_perSample and np.max(values)>np.min(values)):
                    values -= np.min(values)
                    values /= np.max(values)-np.min(values)
                else:
                    pass 
                validSamps += 1
                mySum[levelidx,leftOff:rightOff] += values
                myCounts[levelidx,leftOff:rightOff] += 1
    print()
    if(filtering):
        myTotalSum += mySum
        myTotalCounts += myCounts
        print("filtered {} samples of {}".format(filtered, not_filtered+filtered))
    for levelidx in range(len(levels)):
        result = (mySum/myCounts)[levelidx]
        result -= np.min(result)
        result /= np.max(result)-np.min(result)
        for iidx,i in enumerate(np.arange(-(width//2),width//2+1)*20):
            if front_type == "occ2":
                front_type = "fl.occ"
            baroResults[levelidx].append((i, result[iidx], front_type))

print(totalSamps)
print(validSamps)

if(filtering):
    for levelidx in range(len(levels)):
        result = (myTotalSum/myTotalCounts)[levelidx]
        result -= np.min(result)
        result /= np.max(result)-np.min(result)
        for iidx,i in enumerate(np.arange(-(width//2),width//2+1)*20):
            baroResults[levelidx].append((i, result[iidx], "both"))

hPa = [500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000]
for levelidx in range(len(levels)):
    currhPa = hPa[levels[levelidx]]
    df = pd.DataFrame(baroResults[levelidx], columns=["distance from front (km)", "baroclinity".format(currhPa), "front type"])
    fig = px.line(df, x = "distance from front (km)", y = "baroclinity".format(currhPa), color="front type")
    fig.update_layout(
        height = 150,
        width = 400,
        margin = dict(l=0,r=5,t=0,b=0),
        font_family="Serif",
        legend=dict(
            title=None,
            yanchor="top",
            y=0.99,
            xanchor="right",
            x = 0.99,
            bordercolor="Black",
            borderwidth = 1,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    if(filtering):
        if(filter_north):
            fig.write_image(os.path.join(outpath,"level{}_north.pdf".format(currhPa)))
            fig.write_html(os.path.join(outpath,"level{}_north.html".format(currhPa)))
        else:
            fig.write_image(os.path.join(outpath,"level{}_south.pdf".format(currhPa)))
            fig.write_html(os.path.join(outpath,"level{}_south.html".format(currhPa)))


    else:
        fig.write_image(os.path.join(outpath,"level{}.pdf".format(currhPa)))
        fig.write_html(os.path.join(outpath,"level{}.html".format(currhPa)))
