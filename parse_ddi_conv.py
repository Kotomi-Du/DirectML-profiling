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

    line =["model_name", "kernel_name","input_shape_n","input_shape_c","input_shape_h","input_shape_w", "input_layout", "input_datatype","input_flag","input_padding",\
            "filter_shape_n","filter_shape_c","filter_shape_h","filter_shape_w", "filter_layout", "filter_datatype","filter_flag", "filter_stride_h", "filter_stride_w", "filter_stride_c", "filter_dilation_h", "filter_dilation_w", "filter_dilation_c", "filter_groupcount",
            "output_shape_n","output_shape_c","output_shape_h","output_shape_w", "output_layout", "output_datatype","output_flag", "output_padding", 
            "bias", "bias_flag","direction", "activation" ,"exec_flag", "conv_count"]
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

            filter_datatype = re.findall(r'\((.*?)\)', lines[kernel_name_idx+24])[0]
            filter_flag =  re.findall(r'\((.*?)\)', lines[kernel_name_idx+25])[0]
            filter_shape = []
            for j in range(4):
                filter_shape.append(int(lines[kernel_name_idx+27 + j].rstrip().split("=")[-1],16))
            #layout
            outputdesc_idx = 0
            bias_value = ""
            if "IsNull" in lines[kernel_name_idx+42]:
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
            if "Function" in lines[outputdesc_idx +  41]:
                activation =  re.findall(r'\((.*?)\)', lines[outputdesc_idx + 42])[0]
                exec_flag_idx = outputdesc_idx +  27 # to-do: need to fix
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
            matches = re.findall(r'\((.*?)\)', lines[exec_flag_idx + 5])
            if len(matches) > 1:
                layout_str = matches[1].split(',')
            #print(kernel_name_idx, lines[dim_idx], lines[layout_idx])

            #print(lines[outputdesc_idx+1],layout_str,lines[outputdesc_idx+2])
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

            case_hash = ",".join(str(i) for i in [basename, kernel_name, input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_layout, input_datatype, input_flag,input_padding_str,\
                              filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3], filter_layout, filter_datatype, filter_flag, filter_stride[0], filter_stride[1],filter_stride[2],filter_dilation[0],filter_dilation[1],filter_dilation[2], filter_groupcount, \
                                output_shape[0],output_shape[1],output_shape[2],output_shape[3], output_layout, output_datatype, output_flag, output_padding_str,
                                bias, bias_flag,direction, activation, exec_flag])
            if case_hash not in case_dic.keys():
                case_dic[case_hash] = 1
            else:
                case_dic[case_hash] += 1
            

            
        #     if generate_cmd_flag :
        #         shape_a = ",".join([str(i) for i in inputA_shape])
        #         shape_b = ",".join([str(i) for i in inputB_shape])
        #         datatype = "fp32" if inputA_datatype.strip()=="FLOAT32" else "fp16"
        #         substr_b_info = " --b_managed" if inputB_flag.strip() == "MANAGED" else ""
        #         substr_b_info += " --b_transposed" if transB == "true" else ""
        #         substr_c_info = ""
        #         if inputC_flag != "isnull":
        #             substr_c_info = " --shape_c " + ",".join([str(i) for i in inputC_shape])
        #             if inputC_flag.strip() ==  "MANAGED":
        #                 substr_c_info += " --c_managed"
        #         commandline = ".\cross_runner.exe --type=gemm_dml --iters=1 gemm_opts --gemm_type ab --data_type "+ datatype + "  --layout nchw --shape_a " + shape_a + " --shape_b " +shape_b + substr_b_info + substr_c_info
        #         if commandline not in commandline_set:
        #             commandline_set.add(commandline)
        # corner_case_gemm(lines, i, gemmcase_dic)
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

root_path = r"C:\Users\GAME\Documents\Project\helpWindow\onednn_lnl\SDXL"
basename = "convnchw.log"
commandline_flag = True
create_ddiConv_csv(root_path, basename, commandline_flag)