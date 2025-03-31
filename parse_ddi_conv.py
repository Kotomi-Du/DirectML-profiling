import json
import csv
import re
import pandas as pd
import os

def create_ddiConv_csv(root_path, basename, generate_cmd_flag = False):
    ddi_file = os.path.join(root_path, basename)
    csv_file = os.path.join(root_path, basename.replace(".log","_conv.csv"))
    openf =  open(ddi_file, "r")
    
    csvf = open(csv_file,"w",newline='')
    writer =csv.writer(csvf)

    line =["model_name", "kernel_name","input_shape_n","input_shape_c","input_shape_h","input_shape_w", "input_layout", "input_datatype","input_flag","input_stride","input_padding",\
            "filter_shape_n","filter_shape_c","filter_shape_h","filter_shape_w", "filter_layout", "filter_datatype","filter_flag", "filter_stride_h", "filter_stride_w", "filter_stride_c", "filter_dilation_h", "filter_dilation_w", "filter_dilation_c", "filter_groupcount",
            "output_shape_n","output_shape_c","output_shape_h","output_shape_w", "output_layout", "output_datatype","output_flag", "output_padding", 
            "bias", "bias_flag","direction", "activation" ,"exec_flag", "commandline", "conv_count"]
    writer.writerow(line)
    case_dic = {}
    commandline_set = set()

    lines = openf.readlines()
    #print(len(lines))
    i = -1
    while i < len(lines)-1:
        i+=1
        if "Passed-Metacommand type : Convolution1" in lines[i]:
            #print(lines[i-12:i])
            mc_type = lines[i].rstrip().split("type :")[-1]
            
            kernel_name_idx = i 
            kernel_name = "Convolution1"

            input_datatype = re.findall(r'\((.*?)\)', lines[kernel_name_idx+5])[0]
            input_flag =  re.findall(r'\((.*?)\)', lines[kernel_name_idx+6])[0]
            input_shape = []
            for j in range(4):
                input_shape.append(int(lines[kernel_name_idx+8 + j].rstrip().split("=")[-1],16))

            input_stride = []
            for j in range(3):
                input_stride.append(int(lines[kernel_name_idx+12+j].rstrip().split("=")[-1],16))
            input_stride_str = ";".join([str(i) for i in input_stride])
            filter_datatype = re.findall(r'\((.*?)\)', lines[kernel_name_idx+24])[0]
            filter_flag =  re.findall(r'\((.*?)\)', lines[kernel_name_idx+25])[0]
            filter_shape = []
            for j in range(4):
                filter_shape.append(int(lines[kernel_name_idx+27 + j].rstrip().split("=")[-1],16))
            #layout
            outputdesc_idx = 0
            bias_value = ""
            bias_flag = ""
            print(lines[kernel_name_idx+43])
            if "IsNull" in lines[kernel_name_idx+43]:
                bias_value = "isnull"
                outputdesc_idx = i+5+38+2
            else:
                outputdesc_idx = kernel_name_idx + 62
                bias_datatype = re.findall(r'\((.*?)\)', lines[kernel_name_idx+43])[0]
                bias_flag =  re.findall(r'\((.*?)\)', lines[kernel_name_idx+44])[0]
                bias_shape = []
                for j in range(4):
                    bias_shape.append(int(lines[kernel_name_idx+46 + j].rstrip().split("=")[-1],16))
                bias_value = ";".join([str(i) for i in bias_shape])
            
            output_datatype = re.findall(r'\((.*?)\)', lines[outputdesc_idx+1])[0]
            output_flag =  re.findall(r'\((.*?)\)', lines[outputdesc_idx+2])[0]
            output_shape = []
            for j in range(4):
                output_shape.append(int(lines[outputdesc_idx+4 + j].rstrip().split("=")[-1],16))
                
            direction = re.findall(r'\((.*?)\)', lines[outputdesc_idx+20])[0]

            # inputpadding_list = 
            output_padding = []
            for j in range(5):
                output_padding.append(lines[outputdesc_idx+35+j].rstrip().split("=")[-1])
            output_padding_str = ";".join([str(i) for i in output_padding])
 
            filter_stride = []
            for j in range(3):
                filter_stride.append(int(lines[outputdesc_idx+22+j].rstrip().split("=")[-1],16))

            filter_dilation =  []
            for j in range(3):
                filter_dilation.append(int(lines[outputdesc_idx+25+j].rstrip().split("=")[-1],16))
            
            input_padding = []
            for j in range(6):
                input_padding.append(int(lines[outputdesc_idx+28+j].rstrip().split("=")[-1],16))
            input_padding_str = ";".join([str(i) for i in input_padding])
            
            filter_groupcount =int( re.findall(r'0x\w+', lines[outputdesc_idx+40])[0],16)
            exec_flag_idx = 0
            if "Function" in lines[outputdesc_idx +  42]:
                activation =  re.findall(r'\((.*?)\)', lines[outputdesc_idx + 42])[0]
                exec_flag_idx = outputdesc_idx +  46 # to-do: need to fix
            else:
                activation = "isnull"
                exec_flag_idx = outputdesc_idx + 44
            
            exec_flag_info = re.findall(r'0x\w+', lines[exec_flag_idx])
            if len(exec_flag_info) > 0:
                exec_flag = int( exec_flag_info[0],16)
            else:
                exec_flag = 0

            layout_str = ""
            #print(lines[layout_idx].rstrip(), layout_idx,kernel_name_idx)
            if exec_flag_idx + 5 < len(lines):

                matches = re.findall(r'\((.*?)\)', lines[exec_flag_idx + 5])
                if len(matches) > 1:
                    layout_str = matches[1].split(',')
            else:
                layout_str = ["isnull", "isnull", "isnull"]
            #print(kernel_name_idx, lines[dim_idx], lines[layout_idx])

            info_dic ={}
            input_layout, filter_layout, output_layout = layout_str
            input_datatype = re.findall(r'\((.*?)\)', lines[i+5])[0]
            input_flag = re.findall(r'\((.*?)\)', lines[i+6])[0]
            filter_datatype = re.findall(r'\((.*?)\)', lines[i+5 + 19])[0]
            filter_flag = re.findall(r'\((.*?)\)', lines[i+6+19])[0]
            output_datatype = re.findall(r'\((.*?)\)', lines[outputdesc_idx+1])[0]
            output_flag = re.findall(r'\((.*?)\)', lines[outputdesc_idx+2])[0]
            bias = bias_value
            direction = re.findall(r'\((.*?)\)', lines[outputdesc_idx+20])[0]

            onednn_flag = "false"
            more_info = ""
            for idx in range(exec_flag_idx,exec_flag_idx+14):
                if idx > len(lines)-1:
                    break
                if "OneDNNL can_use_onednn is true" in lines[idx]:
                    onednn_flag = "true"
                    break
            if onednn_flag == "false":
                for idx in range(exec_flag_idx,exec_flag_idx+10):
                    if idx > len(lines)-1:
                        break
                    if "Passed-Metacommand type" in lines[idx]:
                        break
                    if "support" in lines[idx]:
                        clean_info = lines[idx].split(":")[-1].replace(","," ")
                        more_info += clean_info
                    if "Conv Kernel:" in lines[idx]:
                        kernel_name += " " +lines[idx].rstrip().split("Conv Kernel:")[-1].strip()
            else:
                kernel_name += " oneDNN Conv"

            case_hash = ",".join(str(i) for i in [basename, kernel_name, input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_layout, input_datatype, input_flag,input_stride_str,input_padding_str,\
                              filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3], filter_layout, filter_datatype, filter_flag, filter_stride[0], filter_stride[1],filter_stride[2],filter_dilation[0],filter_dilation[1],filter_dilation[2], filter_groupcount, \
                                output_shape[0],output_shape[1],output_shape[2],output_shape[3], output_layout, output_datatype, output_flag, output_padding_str,
                                bias, bias_flag,direction, activation, exec_flag])
                  
            if generate_cmd_flag :
                input_shape = ",".join([str(i) for i in input_shape])
                filter_shape = ",".join([str(i) for i in filter_shape])
                filter_stride = ",".join([str(i) for i in filter_stride])
                datatype = "fp32" if input_datatype.strip()=="FLOAT32" else "fp16"
                bias_flag = str(1) if bias_value.strip() == "isnull" else str(0)
                activation_id = "0"
                if "LEAKY_RELU" in activation:
                    activation_id = "2"
                elif "RELU" in activation:
                    activation_id = "1"
                commandline = ".\cross_runner.exe --type=conv_dml --volatile_flag=0 --iters=10 --no_conform=1 conv_opts --input_shape=" + input_shape + " --filter_shape=" + filter_shape + " --in_pad=1 --out_pad=0 --stride=1,1,1,1 --data_type=" + datatype + " --input_layout=" + input_layout.strip() + " --output_layout=" + output_layout.strip() + " --no_bias=" + bias_flag + " --activation=" + activation_id + " --managed_weights --dnnl_reference"
                if commandline not in commandline_set:
                    commandline_set.add(commandline)
            case_hash +="," + commandline
            if case_hash not in case_dic.keys():
                case_dic[case_hash] = 1
            else:
                case_dic[case_hash] += 1
            

    total_count = 0
    for key, value in case_dic.items():
        row = key.split(",")
        row.append(str(value))
        writer.writerow(row)
        total_count += value
    writer.writerow(["total conv count", str(total_count)])

    if len(commandline_set) > 0:
        for item in commandline_set:
            print(item)
            #writer.writerow(item)
    csvf.close()

root_path = r"C:\Users\yarudu\Documents\Profiling\Adobe\PS"
basename = "DDI_conv.log"
commandline_flag = True
create_ddiConv_csv(root_path, basename, commandline_flag)