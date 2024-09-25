import json
import csv
import re
import pandas as pd
import os



def create_ddiQGEMM_csv(root_path, basename, generate_cmd_flag = False):
    ddi_file = os.path.join(root_path, basename)
    csv_file = os.path.join(root_path, basename.replace(".log","_gemm.csv"))
    openf =  open(ddi_file, "r")
    
    csvf = open(csv_file,"w",newline='')
    writer =csv.writer(csvf)

    line = ["kernel_name","inputA_shape_batch","inputA_shape_channel","inputA_shape_M","inputA_shape_K", "inputA_datatype","inputA_flag","transA",\
       "inputB_shape_batch","inputB_shape_channle","inputB_shape_K","inputB_shape_N", "inputB_datatype","inputB_flag","transB",\
        "inputC_flag", "inputC_broadcast","alpha","beta",\
             "attribute_precision", "activation","exec_flag","case_count"]
    writer.writerow(line)
    gemmcase_dic = {}
    commandline_set = set()

    lines = openf.readlines()
    #print(len(lines))
    i = -1
    while i < len(lines)-1:
        i+=1
        if "Passed-Metacommand type : QUANTIZED_GEMM" in lines[i]:
            kernel_name_idx = i
            kernel_name = ""
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
            inputC_stride = []
            inputC_broadcast = "False"
            if "IsNull" in lines[kernel_name_idx + 43]:
                inputC_flag = "isnull"
                outputdesc_idx = kernel_name_idx + 127
            else:
                inputC_flag = re.findall(r'\((.*?)\)', lines[kernel_name_idx + 44])[0]
                for j in range(4):
                    inputC_shape.append(int(lines[kernel_name_idx+46 +j].rstrip().split("=")[-1],16))
                for j in range(4):
                    inputC_stride.append(int(lines[kernel_name_idx+50 +j].rstrip().split("=")[-1],16))
                if inputC_stride[2] == 0 and inputC_shape[2] != 1:
                    inputC_broadcast = "True"
                    print(kernel_name_idx, lines[kernel_name_idx+144 ])
                outputdesc_idx = kernel_name_idx + 144
            
            attribute_precision = lines[outputdesc_idx +  19].rstrip().split("=")[-1]
            transA = "false" if int(lines[outputdesc_idx +  20].rstrip().split("=")[-1],16) == 0 else "true"
            transB = "false" if int(lines[outputdesc_idx +  21].rstrip().split("=")[-1],16) ==0 else "true"

            alpha = lines[outputdesc_idx +  22].rstrip().split("=")[-1]
            beta = lines[outputdesc_idx +  23].rstrip().split("=")[-1]
            #print( lines[kernel_name_idx +  64])
            
            exec_flag_idx = 0
            if "Function" in lines[outputdesc_idx +  25]:
                activation =  re.findall(r'\((.*?)\)', lines[outputdesc_idx + 25])[0]
                exec_flag_idx = outputdesc_idx +  30 # to-do: need to fix
            else:
                activation = "isnull"
                exec_flag_idx = outputdesc_idx + 31

            exec_flag_info = re.findall(r'0x\w+', lines[exec_flag_idx])
            if len(exec_flag_info) > 0:
                exec_flag = int( exec_flag_info[0],16)
            else:
                exec_flag = 0
       
            #print(lines[outputdesc_idx +  24])
            for idx in range(exec_flag_idx,exec_flag_idx+14):
                if idx > len(lines)-1:
                    break
                if "QuantizedGEMM Shader Code =" in lines[idx]:
                    kernel_name = lines[idx].rstrip().split("QuantizedGEMM Shader Code =")[-1].strip()
                    break
                if "Unsupported parameters of quantized gemm" in lines[idx]:
                    kernel_name = "no kernel available"
                    break
    
           
            if exec_flag == 0 :
                print(lines[exec_flag_idx],exec_flag_idx)
            
            case_hash = ",".join(str(i) for i in [ kernel_name, inputA_shape[0], inputA_shape[1],inputA_shape[2], inputA_shape[3], inputA_datatype, inputA_flag,transA,\
                              inputB_shape[0], inputB_shape[1],inputB_shape[2], inputB_shape[3], inputB_datatype, inputB_flag,transB, \
                                 inputC_flag, inputC_broadcast, alpha, beta, \
                                 attribute_precision,activation, exec_flag])
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
    writer.writerow(["total QGEMM count", str(total_gemm_count)])

    if len(commandline_set) > 0:
        for item in commandline_set:
            print(item)
            #writer.writerow(item)
    csvf.close()

root_path = r"C:\Users\yarudu\Downloads\oneDNN_BMG"
basename = "phi3mini_1024_256 1.log"
commandline_flag = True
create_ddiQGEMM_csv(root_path, basename, commandline_flag)