import json
import csv
import re
import pandas as pd
import os

def create_ddiGEMM_csv(root_path, basename):
    ddi_file = os.path.join(root_path, basename)
    csv_file = os.path.join(root_path, basename.replace(".log","_gemm.csv"))
    openf =  open(ddi_file, "r")
    
    csvf = open(csv_file,"w",newline='')
    writer =csv.writer(csvf)

    line = ["model_name", "kernel_name","inputA_shape_B","inputA_shape_C","inputA_shape_M","inputA_shape_K", "inputA_datatype","inputA_flag","transA",\
       "inputB_shape_B","inputB_shape_C","inputB_shape_K","inputB_shape_N", "inputB_datatype","inputB_flag","transB",\
        "inputC_flag", "alpha","beta",\
             "attribute_precision", "activation","exec_flag","use_onednn", "zero_pool_memory_size","case_count"]
    writer.writerow(line)
    gemmcase_dic = {}

    lines = openf.readlines()
    #print(len(lines))
    i = -1
    while i < len(lines)-1:
        i+=1
        if "Passed-Metacommand type : GEMM1" in lines[i]:
            info_dic ={}
            mc_type = lines[i].rstrip().split("type :")[-1]
            kernel_name_idx = i
            kernel_name = "OneDNN"
            inputA_datatype = re.findall(r'\((.*?)\)', lines[kernel_name_idx+5])[0]
            inputA_flag =  re.findall(r'\((.*?)\)', lines[kernel_name_idx+6])[0]
            inputA_shape = []
            for j in range(4):
                inputA_shape.append(int(lines[kernel_name_idx+8 + j].rstrip().split("=")[-1],16))

            inputB_datatype = re.findall(r'\((.*?)\)', lines[kernel_name_idx+24])[0]
            inputB_flag = re.findall(r'\((.*?)\)', lines[kernel_name_idx + 25])[0]
            inputB_shape = []
            for j in range(4):
                inputB_shape.append(int(lines[kernel_name_idx+27 +j].rstrip().split("=")[-1],16))
           
            inputC_flag = ""
            if "IsNull" in lines[kernel_name_idx + 43]:
                inputC_flag = "isnull"
                outputdesc_idx = kernel_name_idx + 45
            else:
                inputC_flag = "has_value"
                outputdesc_idx = kernel_name_idx + 62
            transA = "false" if int(lines[outputdesc_idx +  20].rstrip().split("=")[-1],16) == 0 else "true"
            transB = "false" if int(lines[outputdesc_idx +  21].rstrip().split("=")[-1],16) ==0 else "true"
            alpha = lines[outputdesc_idx +  22].rstrip().split("=")[-1]
            beta = lines[outputdesc_idx +  23].rstrip().split("=")[-1]
            #print( lines[kernel_name_idx +  64])
            attribute_precision = lines[outputdesc_idx +  19].rstrip().split("=")[-1]
            
            exec_flag_idx = 0
            if "Function" in lines[outputdesc_idx +  25]:
                activation =  re.findall(r'\((.*?)\)', lines[kernel_name_idx + 70])[0]
                exec_flag_idx = outputdesc_idx +  27 # to-do: need to fix
            else:
                activation = "isnull"
                exec_flag_idx = outputdesc_idx + 28

            exec_flag_info = re.findall(r'0x\w+', lines[exec_flag_idx])
            if len(exec_flag_info) > 0:
                exec_flag = int( exec_flag_info[0],16)
            else:
                exec_flag = 0
            zero_pool_memory_size = 0
            onednn_flag = "false"
            #print(lines[outputdesc_idx +  24])
            for idx in range(exec_flag_idx,exec_flag_idx+14):
                if "OneDNNL can_use_onednn is true" in lines[idx]:
                    onednn_flag = "true"
                if "OneDNN GEMMs with zero pool is disabled" in  lines[idx]:
                    log = lines[idx].rstrip()
                    zero_pool_memory_size = re.findall(r'\d+', log)[-1]
                    break

            case_hash = ",".join(str(i) for i in [basename, kernel_name, inputA_shape[0], inputA_shape[1],inputA_shape[2], inputA_shape[3], inputA_datatype, inputA_flag,transA,\
                              inputB_shape[0], inputB_shape[1],inputB_shape[2], inputB_shape[3], inputB_datatype, inputB_flag,transB, \
                                 inputC_flag,alpha, beta, \
                                 attribute_precision,activation, exec_flag, onednn_flag, zero_pool_memory_size])
            if case_hash not in gemmcase_dic.keys():
                gemmcase_dic[case_hash] = 1
            else:
                gemmcase_dic[case_hash] += 1
    total_gemm_count = 0
    for key, value in gemmcase_dic.items():
        row = key.split(",")
        row.append(str(value))
        writer.writerow(row)
        total_gemm_count += value
    writer.writerow(["total gemm count", str(total_gemm_count)])
    csvf.close()

root_path = r"C:\Users\GAME\Documents\Project\helpWindow\onednn_lnl"
basename = "unet_broadcast.log"
create_ddiGEMM_csv(root_path, basename)