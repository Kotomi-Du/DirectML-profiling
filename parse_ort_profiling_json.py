# index value in profile is same with allocator_planner.cc
# index value in tensor allocation is different, where is it?
import csv
import json
time_file = r'C:\Users\GAME\Documents\Project\INT8\mobilenetv2-12-int8\mobilenet_dml.json_2023-12-12_17-04-08.json'
csv_file = time_file.replace("json", 'csv')
csvf = open(csv_file,"w",newline='')
writer = csv.writer(csvf)
 # only has three inputs and  one output as defualt, may need to change as needed
line = ["name","op_type", "input1_type", "input1_shape", "input2_type", "input2_shape","input3_type", "input3_shape","output_type","output_shape","duration"]
writer.writerow(line)
count = 0
node_count =0
total_duration = 0
count_time={}
count_name = {}
other_info={}
with open(time_file, 'r') as f:
    # load the contents of the file into a dictionary
    data = json.load(f)
    for i in range(len(data)):
        # if "/conv_in/Conv_fence_before" in data[i]["name"]:
        #     count += 1
        # if count == 2:
        #     break
        if "kernel" in data[i]["name"]: 
            output_name = data[i]["name"].split("_kernel")[0]
            if output_name not in count_time:
                count_time[output_name] = int(data[i]["dur"])
                count_name[output_name] = 1

                
                op_type = data[i]['args']['op_name']
                
                output_size = data[i]['args']['output_size']
                input_type = ["none"]*3
                input_shape = [[]] * 3
                output_type = None
                output_shape = None
                for j, p in enumerate(data[i]['args']['input_type_shape']):
                    if j == 3:
                        break
                    for key, value in p.items():
                        input_type[j]= key
                        input_shape[j]= value
                for j, p in enumerate(data[i]['args']['output_type_shape']):
                    for key, value in p.items():
                        output_type = key
                        output_shape = value
                        break
                    break

                other_info[output_name] = [output_name, op_type, input_type[0],input_shape[0],\
                                            input_type[1],input_shape[1],
                                                input_type[2], input_shape[2],
                                                output_type,output_shape]
            else:
                count_time[output_name] += int(data[i]["dur"])
                count_name[output_name] += 1


for key, value in count_time.items():
    duration = round(value / count_name[key],2)
    total_duration+=duration
    kernel_info = other_info[key]
    
    kernel_info.append(duration)
    writer.writerow(kernel_info)
    node_count+=1
csvf.close()
print("total duration:",round(total_duration/1000,2))
print("Done to generate file {}". format(csv_file))