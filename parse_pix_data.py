import os
import csv
import pandas as pd
rootpath = r"C:\Users\GAME\Documents\Project\helpWindow\onednn_lnl"
log_file = "pix_lnl_llama_2048token.txt"
file = os.path.join(rootpath, log_file)
file = file.strip(".txt")
rawdata = pd.read_csv(f"{file}.txt",delimiter = '\t')

signal_count = 0
first_iteration_start = 0
second_iteration_start = 0  
## Option 1:
#  use "DML_EXECUTION_PLAN" to find where is the start of first_iteration and second iteration
#  value = line number - 1
#  
## Option 2:
#  use the algorithm below to find first iteration and second iteration
#  but most of the time it does not work
'''
for index, line in rawdata.iterrows():
    if signal_count == 5:
        first_iteration = index
    if signal_count == 10:
        second_iteration = index
        break
    if "Signal" in line :
        signal_count +=0
'''
## Option 3: [todo] use the whole information to decide

first_iteration_start = 75
second_iteration_start = 913
prevline = ""
ex_operator_list=[]
ex_time = []

dispatch_operator_list=[]
dispatch_time=[]

pre_checkDispatch =False
idx = -1
while idx < len(rawdata)-1: 
    idx+=1
    # if idx < first_iteration_start:
    #     continue
    # if idx > second_iteration_start:
    #     break
    line = rawdata.iloc[idx]
    if "ExecuteMetaCommand" in line[2]:    
        ex_operator_list.append((prevline[2].strip()))
        ex_time.append(int(str(prevline[-1]).replace(",","")))
        # temp.append(line)
    if "Dispatch" in line[2]:
        if pre_checkDispatch:
            continue
        #print(line[2])
        dispatch_operator_list.append((prevline[2].strip()))
        dispatch_time.append(int(str(prevline[-1]).replace(",","")))
        pre_checkDispatch = True 
    else:
        prevline = line
        pre_checkDispatch = False

sumup= (sum(dispatch_time) + sum(ex_time))/1000000
print("total latency per iteration: {} ms \n \
      Tip: if the data is too different from ort_perf_test.exe,\n \
      please double check the first/second iteration number".format(round(sumup,2)))


csv_file = f"{file}_test.csv"
csvf = open(csv_file,"w",newline='')
writer = csv.writer(csvf)
line = ["execute type","layer type","layer name","time"]
writer.writerow(line)

for op,time in zip(ex_operator_list,ex_time):
    layer_info = op.split(",")[-1]
    if "(" in layer_info:
        mark_idx = layer_info.index("(")
        layer_type = layer_info[0:mark_idx-1]
        layer_name = layer_info[mark_idx+1:-1]
    else:
        mark_idx = 0
        layer_type = layer_info[0:]
        layer_name = "unknown"
    writer.writerow(["ExecuteMetaCommand",layer_type, layer_name,round(float(time)/1000000,2)])

for op,time in zip(dispatch_operator_list,dispatch_time):
    layer_info = op.split(",")[-1]
    if "(" in layer_info:
        mark_idx = layer_info.index("(")
        layer_type = layer_info[0:mark_idx-1]
        layer_name = layer_info[mark_idx+1:-1]
    else:# some op may not have infomation in (), e.g.DML_OPERATOR_ACTIVATION_GELU
        mark_idx = 0
        layer_type = layer_info[0:]
        layer_name = "unknown"
    writer.writerow(["Dispatch",layer_type, layer_name,round(float(time)/1000000,2)])


csvf.close()
print("{} generated".format(csv_file))

import pandas as pd
df1 = pd.read_csv(os.path.join(rootpath, csv_file))
filtered_df1 = df1[(df1['layer type'] == 'DML_OPERATOR_CONVOLUTION') & (df1['execute type'] == 'ExecuteMetaCommand')]
filtered_df1.to_csv(os.path.join(rootpath, 'temp.csv'), index=False)