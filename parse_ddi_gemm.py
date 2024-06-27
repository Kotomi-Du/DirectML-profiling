import json
import csv
import re
import pandas as pd
import os

def corner_case_gemm(lines,i, gemmcase_dic):
    # this looks like from a old spec
    if "Passed-Metacommand type : GEMM" in lines[i] and "GEMM1" not in lines[i]:
            info_dic ={}
            mc_type = lines[i].rstrip().split("type :")[-1]
            kernel_name_idx = i
            kernel_name = "GEMM"
            inputA_datatype = re.findall(r'\((.*?)\)', lines[kernel_name_idx+5])[0]
            inputA_flag =  re.findall(r'\((.*?)\)', lines[kernel_name_idx+7])[0]
            inputA_shape = []
            for j in range(4):
                inputA_shape.append(int(lines[kernel_name_idx+9 + j].rstrip().split("=")[-1],16))

            inputB_datatype = re.findall(r'\((.*?)\)', lines[kernel_name_idx+19])[0]
            inputB_flag = re.findall(r'\((.*?)\)', lines[kernel_name_idx + 21])[0]
            inputB_shape = []
            for j in range(4):
                inputB_shape.append(int(lines[kernel_name_idx+23 +j].rstrip().split("=")[-1],16))
           
            inputC_flag = ""
            if "IsNull" in lines[kernel_name_idx + 33]:
                inputC_flag = "isnull"
                outputdesc_idx = kernel_name_idx + 32
            else:
                inputC_flag = "has_value"
                outputdesc_idx = kernel_name_idx + 48
            transA = "false" if int(lines[outputdesc_idx +  14].rstrip().split("=")[-1],16) == 0 else "true"
            transB = "false" if int(lines[outputdesc_idx +  15].rstrip().split("=")[-1],16) ==0 else "true"
            alpha = lines[outputdesc_idx +  16].rstrip().split("=")[-1]
            beta = lines[outputdesc_idx +  17].rstrip().split("=")[-1]
            #print( lines[kernel_name_idx +  64])
            attribute_precision = lines[outputdesc_idx +  13].rstrip().split("=")[-1]
            
            exec_flag_idx = 0
            if "Function" in lines[outputdesc_idx +  19]:
                activation =  re.findall(r'\((.*?)\)', lines[kernel_name_idx + 70])[0]
                exec_flag_idx = outputdesc_idx +  27 # to-do: need to fix
            else:
                activation = "isnull"
                exec_flag_idx = outputdesc_idx + 21
            #print(lines[exec_flag_idx])
            exec_flag_info = re.findall(r'0x\w+', lines[exec_flag_idx])
            if len(exec_flag_info) > 0:
                exec_flag = int( exec_flag_info[0],16)
            else:
                exec_flag = 0
            zero_pool_memory_size = 0
            onednn_flag = "false"
            more_info = ""
            #print(lines[outputdesc_idx +  24])
            for idx in range(exec_flag_idx,exec_flag_idx+14):
                if idx > len(lines)-1:
                    break
                if "OneDNNL can_use_onednn is true" in lines[idx]:
                    onednn_flag = "true"
                if "OneDNN GEMMs with zero pool is disabled" in  lines[idx]:
                    log = lines[idx].rstrip()
                    zero_pool_memory_size = re.findall(r'\d+', log)[-1]
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
                    if "Gemm Kernel:" in lines[idx]:
                        kernel_name += " " +lines[idx].rstrip().split("Gemm Kernel:")[-1].strip()
            else:
                kernel_name += " oneDNN GEMM"

            case_hash = ",".join(str(i) for i in [basename, kernel_name, inputA_shape[0], inputA_shape[1],inputA_shape[2], inputA_shape[3], inputA_datatype, inputA_flag,transA,\
                              inputB_shape[0], inputB_shape[1],inputB_shape[2], inputB_shape[3], inputB_datatype, inputB_flag,transB, \
                                 inputC_flag,alpha, beta, \
                                 attribute_precision,activation, exec_flag, onednn_flag, zero_pool_memory_size,more_info])
            if case_hash not in gemmcase_dic.keys():
                gemmcase_dic[case_hash] = 1
            else:
                gemmcase_dic[case_hash] += 1
            

def create_ddiGEMM_csv(root_path, basename, generate_cmd_flag = False):
    ddi_file = os.path.join(root_path, basename)
    csv_file = os.path.join(root_path, basename.replace(".log","_gemm.csv"))
    openf =  open(ddi_file, "r")
    
    csvf = open(csv_file,"w",newline='')
    writer =csv.writer(csvf)

    line = ["model_name", "kernel_name","inputA_shape_B","inputA_shape_C","inputA_shape_M","inputA_shape_K", "inputA_datatype","inputA_flag","transA",\
       "inputB_shape_B","inputB_shape_C","inputB_shape_K","inputB_shape_N", "inputB_datatype","inputB_flag","transB",\
        "inputC_flag", "alpha","beta",\
             "attribute_precision", "activation","exec_flag","use_onednn", "zero_pool_memory_size", "more_info" ,"case_count"]
    writer.writerow(line)
    gemmcase_dic = {}
    commandline_set = set()

    lines = openf.readlines()
    #print(len(lines))
    i = -1
    while i < len(lines)-1:
        i+=1
        if "Passed-Metacommand type : GEMM1" in lines[i]:
            info_dic ={}
            mc_type = lines[i].rstrip().split("type :")[-1]
            kernel_name_idx = i
            kernel_name = "GEMM1"
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
            inputC_shape = []
            if "IsNull" in lines[kernel_name_idx + 43]:
                inputC_flag = "isnull"
                outputdesc_idx = kernel_name_idx + 45
            else:
                inputC_flag = re.findall(r'\((.*?)\)', lines[kernel_name_idx + 44])[0]
                for j in range(4):
                    inputC_shape.append(int(lines[kernel_name_idx+46 +j].rstrip().split("=")[-1],16))
                outputdesc_idx = kernel_name_idx + 62
            transA = "false" if int(lines[outputdesc_idx +  20].rstrip().split("=")[-1],16) == 0 else "true"
            transB = "false" if int(lines[outputdesc_idx +  21].rstrip().split("=")[-1],16) ==0 else "true"
            alpha = lines[outputdesc_idx +  22].rstrip().split("=")[-1]
            beta = lines[outputdesc_idx +  23].rstrip().split("=")[-1]
            #print( lines[kernel_name_idx +  64])
            attribute_precision = lines[outputdesc_idx +  19].rstrip().split("=")[-1]
            
            exec_flag_idx = 0
            if "Function" in lines[outputdesc_idx +  25]:
                activation =  re.findall(r'\((.*?)\)', lines[outputdesc_idx + 25])[0]
                exec_flag_idx = outputdesc_idx +  27 # to-do: need to fix
            else:
                activation = "isnull"
                exec_flag_idx = outputdesc_idx + 27
            #print(lines[exec_flag_idx])
            exec_flag_info = re.findall(r'0x\w+', lines[exec_flag_idx])
            if len(exec_flag_info) > 0:
                exec_flag = int( exec_flag_info[0],16)
            else:
                exec_flag = 0
            zero_pool_memory_size = 0
            onednn_flag = "false"
            more_info = ""
            #print(lines[outputdesc_idx +  24])
            for idx in range(exec_flag_idx,exec_flag_idx+14):
                if idx > len(lines)-1:
                    break
                if "OneDNNL can_use_onednn is true" in lines[idx]:
                    onednn_flag = "true"
                if "OneDNN GEMMs with zero pool is disabled" in  lines[idx]:
                    log = lines[idx].rstrip()
                    zero_pool_memory_size = re.findall(r'\d+', log)[-1]
                    break
            if onednn_flag == "false":
                for idx in range(exec_flag_idx,exec_flag_idx+10):
                    if "Passed-Metacommand type" in lines[idx]:
                        break
                    if "support" in lines[idx]:
                        clean_info = lines[idx].split(":")[-1].replace(","," ")
                        more_info += clean_info
                    if "Gemm Kernel:" in lines[idx]:
                        kernel_name += " " +lines[idx].rstrip().split("Gemm Kernel:")[-1].strip()
            else:
                kernel_name += " oneDNN GEMM"

            case_hash = ",".join(str(i) for i in [basename, kernel_name, inputA_shape[0], inputA_shape[1],inputA_shape[2], inputA_shape[3], inputA_datatype, inputA_flag,transA,\
                              inputB_shape[0], inputB_shape[1],inputB_shape[2], inputB_shape[3], inputB_datatype, inputB_flag,transB, \
                                 inputC_flag,alpha, beta, \
                                 attribute_precision,activation, exec_flag, onednn_flag, zero_pool_memory_size,more_info])
            if case_hash not in gemmcase_dic.keys():
                gemmcase_dic[case_hash] = 1
            else:
                gemmcase_dic[case_hash] += 1
            
            if generate_cmd_flag :
                shape_a = ",".join([str(i) for i in inputA_shape])
                shape_b = ",".join([str(i) for i in inputB_shape])
                datatype = "fp32" if inputA_datatype.strip()=="FLOAT32" else "fp16"
                substr_b_info = " --b_managed" if inputB_flag.strip() == "MANAGED" else ""
                substr_b_info += " --b_transposed" if transB == "true" else ""
                substr_c_info = ""
                if inputC_flag != "isnull":
                    substr_c_info = " --shape_c " + ",".join([str(i) for i in inputC_shape])
                    if inputC_flag.strip() ==  "MANAGED":
                        substr_c_info += " --c_managed"
                commandline = ".\cross_runner.exe --type=gemm_dml --iters=1 gemm_opts --gemm_type ab --data_type "+ datatype + "  --layout nchw --shape_a " + shape_a + " --shape_b " +shape_b + substr_b_info + substr_c_info
                if commandline not in commandline_set:
                    commandline_set.add(commandline)
        corner_case_gemm(lines, i, gemmcase_dic)
    total_gemm_count = 0
    for key, value in gemmcase_dic.items():
        row = key.split(",")
        row.append(str(value))
        writer.writerow(row)
        total_gemm_count += value
    writer.writerow(["total gemm count", str(total_gemm_count)])

    if len(commandline_set) > 0:
        for item in commandline_set:
            print(item)
            #writer.writerow(item)
    csvf.close()

root_path = r"C:\Users\GAME\Documents\Project\helpWindow\onednn_lnl\SDXL"
basename = "convnchw.log"
commandline_flag = True
create_ddiGEMM_csv(root_path, basename, commandline_flag)