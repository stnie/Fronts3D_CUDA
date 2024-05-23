import numpy as np
import sys
import pandas as pd



# very rough script to aggregate printed timings from code
# input should be the timerRead file. Calc and Write are then automatically added.
files = [sys.argv[1]]
if("Read" in files[0]):
  files.append(files[0].replace("Read","Calc"))
  files.append(files[0].replace("Read","Write"))


allList = []
for file in files:
    with open(file, "r") as f:
        if "Write" in file:
            partname = "write"
        elif "Read" in file:
            partname = "read"
        elif "Calc" in file:
            partname = "calc"
        else:
            partname = "segments"
        for line in f:
            if "TIMING:" in line:
                parts = line.split(" ")
                try:
                    time = parts[1]
                    calcType = parts[3].strip()[1:]
                    convertedTime = float(time)
                    name = " ".join(parts[4:]).strip()[:-1]
                    allList.append([partname, name, calcType, convertedTime])
                except:
                    pass

# print segment size informations 
if partname == "segments":
    myFrame = pd.DataFrame(allList, index=np.arange(len(allList)), columns=["part", "action", "type", "time"])
    myFrame = myFrame.groupby(["action"]).agg({'part': 'first', 'type': 'first', 'time' : 'min'})
    print(myFrame)
    myFrame = pd.DataFrame(allList, index=np.arange(len(allList)), columns=["part", "action", "type", "time"])
    myFrame = myFrame.groupby(["action"]).agg({'part': 'first', 'type': 'first', 'time' : 'max'})
    print(myFrame)
    myFrame = pd.DataFrame(allList, index=np.arange(len(allList)), columns=["part", "action", "type", "time"])
    myFrame = myFrame.groupby(["action"]).agg({'part': 'first', 'type': 'first', 'time' : 'mean'})
    print(myFrame)


myFrame = pd.DataFrame(allList, index=np.arange(len(allList)), columns=["part", "action", "type", "time"])
myFrame = myFrame.groupby(["action"]).agg({'part': 'first', 'type': 'first', 'time' : 'sum'})
myFrame.time = myFrame.time.div(720)

myPartFrame = myFrame[~myFrame.type.str.contains("total")]


# prints timings of individual measurements
print(myPartFrame)
totalCalc = myFrame.loc["process file"].time
totalRead = myFrame.loc["read file"].time
totalWrite = myFrame.loc["write file"].time

aggregatedTimes = myPartFrame.groupby(["part"]).sum()

aggregatedCalc = aggregatedTimes.loc["calc"].time
aggregatedRead = aggregatedTimes.loc["read"].time
aggregatedWrite = aggregatedTimes.loc["write"].time
myPartFrame.loc["calc other"] = ["calc", "other", totalCalc-aggregatedCalc]
myPartFrame.loc["read other"] = ["read", "other", totalRead-aggregatedRead]
myPartFrame.loc["write other"] = ["write", "other", totalWrite-aggregatedWrite]
aggregatedTypes = myPartFrame.groupby(["part", "type"]).sum()

# prints timings of stages
print(aggregatedTypes)
