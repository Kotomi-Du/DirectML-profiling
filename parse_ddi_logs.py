import os
import json
# test for parsing the args of metacommand 
import re
import glob
import pandas as pd
import os
import csv
import shutil


def create_ddiConv_json(ddi_file):
    json_file =  os.path.basename(ddi_file).replace(".log","_conv.json")
    dic ={}
    openf =  open(ddi_file, "r")

    lines = openf.readlines()
    #print(len(lines))
    i = -1
    while i < len(lines)-1:
        i+=1
        if "Passed-Metacommand type : Convolution1" in lines[i]:
            #print(lines[i-12:i])
            mc_type = lines[i].rstrip().split("type :")[-1]
            
            kernel_name_idx = -1
            for k in range(11):
                if "Conv Kernel:" in lines[i-k] \
                    or "Gemm Kernel:" in lines[i-k]:  # for the case of GemmBasedConvolution 
                    kernel_name_idx = i-k
                    # print(lines[kernel_name_idx])
                    # print(kernel_name_idx)
                    break
                if k == 10:
                    assert("no conv kernel")
                    
            kernel_name = lines[kernel_name_idx].rstrip().split("Kernel:")[-1]
            #print(mc_type, kernel_name)
            if mc_type not in dic.keys():
                dic[mc_type] = {}
            if kernel_name not in dic[mc_type].keys():
                dic[mc_type][kernel_name] = []
            
            layout_idx = -1
            dim_idx = -1
            param_str = []
            for k in range(8):
                dim_idx = kernel_name_idx-k
                layout_idx = kernel_name_idx - k+2
                param_str =  re.findall(r'0x\w+', lines[dim_idx])
                if len(param_str) > 5 \
                    and "Filter" in lines[dim_idx]:  # for the case of GemmBasedConvolution
                    break

            # print(dim_idx, lines[dim_idx])   
            # print(param_str)
            param_list = []
            for idx, val in enumerate(param_str):
                param_list.append(int(val, 16))
            
            layout_str = ""
            #print(lines[layout_idx].rstrip(), layout_idx,kernel_name_idx)
            layout_str = re.findall(r'N\w+', lines[layout_idx])
            #print(kernel_name_idx, lines[dim_idx], lines[layout_idx])

            ''' some log  BiasDesc = isNull'''
            output_shape = []
            if "IsNull" in lines[i+5+38]:
                bias_value = "isnull"
                outputdesc_idx = i+5+38+2
                output_shape = param_list[9:13]
            else:
                outputdesc_idx = i+5+19+38
                bias_value = param_list[9:13]
                output_shape =  param_list[13:17]

            input_stride = []
            for j in range(4):
                input_stride.append(int(lines[i+12+j].rstrip().split("=")[-1],16))
            output_stride = []
            for j in range(4):
                output_stride.append(int(lines[outputdesc_idx+8+j].rstrip().split("=")[-1],16))

            # inputpadding_list = 
            outputpadding_list = []
            for j in range(5):
                outputpadding_list.append(lines[outputdesc_idx+35+j].rstrip().split("=")[-1])
 
            filter_stride = []
            for j in range(3):
                filter_stride.append(int(lines[outputdesc_idx+22+j].rstrip().split("=")[-1],16))

            filter_dilation =  []
            for j in range(3):
                filter_dilation.append(int(lines[outputdesc_idx+25+j].rstrip().split("=")[-1],16))
            
            input_padding = []
            for j in range(6):
                input_padding.append(int(lines[outputdesc_idx+28+j].rstrip().split("=")[-1],16))
            

            
            group_count =int( re.findall(r'0x\w+', lines[outputdesc_idx+40])[0],16)
            
            info_dic ={}
            info_dic["input"] = {"shape": param_list[1:5], "layout": layout_str[0], "datatype": re.findall(r'\((.*?)\)', lines[i+5])[0], "flag": re.findall(r'\((.*?)\)', lines[i+6])[0], "stride": input_stride, "padding": input_padding}
            info_dic["filter"] = {"shape": param_list[5:9], "layout": layout_str[1], "datatype": re.findall(r'\((.*?)\)', lines[i+5 + 19])[0],"flag": re.findall(r'\((.*?)\)', lines[i+6+19])[0],"stirde": filter_stride, "dilation":filter_dilation,  "groupcount": group_count}
            info_dic["output"] = {"shape":output_shape, "layout": layout_str[2], "datatype": re.findall(r'\((.*?)\)', lines[outputdesc_idx+1])[0],"flag":re.findall(r'\((.*?)\)', lines[outputdesc_idx+2])[0], "stride": output_stride,"padding":outputpadding_list}
            info_dic["bias"] = bias_value
            info_dic["direction"] = re.findall(r'\((.*?)\)', lines[outputdesc_idx+20])[0]
            
            exec_flag_idx = 0
            if "Function" in lines[outputdesc_idx+42]:
                info_dic["activation"] =  re.findall(r'\((.*?)\)', lines[outputdesc_idx+42])[0]
                exec_flag_idx = outputdesc_idx+46
            else:
                info_dic["activation"] = "isnull"
                exec_flag_idx = outputdesc_idx+44
            
            exec_flag_info = re.findall(r'0x\w+', lines[exec_flag_idx])
            if len(exec_flag_info) > 0:
                info_dic["exec_flag"] = int( exec_flag_info[0],16)
            else:
                info_dic["exec_flag"] = 0
                
            dic[mc_type][kernel_name].append(info_dic)
            
            
        

    json_object = json.dumps(dic, indent=4)
    with open(json_file, "w") as outfile:
        outfile.write(json_object)

def create_ddiGEMM_json(ddi_file):
    json_file =  os.path.basename(ddi_file).replace(".log","_gemm.json")
    dic ={}
    openf =  open(ddi_file, "r")

    lines = openf.readlines()
    #print(len(lines))
    i = -1
    while i < len(lines)-1:
        i+=1
        if "Passed-Metacommand type : GEMM1" in lines[i]:
            info_dic ={}
            mc_type = lines[i].rstrip().split("type :")[-1]
            kernel_name_idx = -1
            kernel_name = ""
            for k in range(8):
                if "Gemm Kernel:" in lines[i-k]:
                    kernel_name_idx = i-k
                    kernel_name = lines[kernel_name_idx].rstrip().split("Kernel:")[-1]
                    break
                if k == 6:
                    kernel_name_idx = i-4
                    kernel_name = "unknown"
                    assert("no gemm kernel")

            #print(mc_type, kernel_name)
            if mc_type not in dic.keys():
                dic[mc_type] = {}
            if kernel_name not in dic[mc_type].keys():
                dic[mc_type][kernel_name] = []
            
            dim_idx = kernel_name_idx - 2
            param_str =  re.findall(r'0x\w+', lines[dim_idx])
            param_list = []
            for idx, val in enumerate(param_str):
                param_list.append(int(val, 16))

            # print(param_list)
            # exit()
            inputA_shape = param_list[3:5] # this first value is for thread id, gemm only has two dim MxK
            inputB_shape = param_list[7:9] # K x N
            
            
            outputdesc_idx = 0
            if "IsNull" in lines[i + 41+ 2]:
                outputdesc_idx = i + 41 + 4 # 4 is start from Cdes
                info_dic["inputC"] = "isnull"
                output_shape = param_list[11:13]
            else:
                # Cdes is not null
                outputdesc_idx = i + 41 + 21
                info_dic["inputC"] = "not null"
                output_shape = param_list[15:17]

            '''stride is useless '''
            # inputA_stride = []
            # for j in range(4):
            #     inputA_stride.append(int(lines[i+12+j].rstrip().split("=")[-1],16))
            
            # inputB_stride = []
            # for j in range(4):
            #     inputB_stride.append(int(lines[i+31+j].rstrip().split("=")[-1],16))
            #     #print(lines[i+31+j])

            # output_stride = []
            # for j in range(4):
            #     output_stride.append(int(lines[outputdesc_idx +  8 +j].rstrip().split("=")[-1],16))
            #     #print(lines[outputdesc_idx +  8 +j])
            
            transA = "false" if int(lines[outputdesc_idx +  20].rstrip().split("=")[-1],16) == 0 else "true"
            transB = "false" if int(lines[outputdesc_idx +  21].rstrip().split("=")[-1],16) ==0 else "true"
            alpha = lines[outputdesc_idx +  22].rstrip().split("=")[-1]
            beta = lines[outputdesc_idx +  23].rstrip().split("=")[-1]

            info_dic["config"] = {"alpha": alpha, "beta": beta}
            info_dic["inputA"] = {"shape": inputA_shape,  "datatype": re.findall(r'\((.*?)\)', lines[i+5])[0], "flag": re.findall(r'\((.*?)\)', lines[i+6])[0], "transA": transA}
            info_dic["inputB"] = {"shape": inputB_shape,  "datatype": re.findall(r'\((.*?)\)', lines[i+5+19])[0], "flag": re.findall(r'\((.*?)\)', lines[i+6 + 19])[0], "transB": transB}       
            info_dic["output"] = {"shape":output_shape,   "datatype": re.findall(r'\((.*?)\)', lines[outputdesc_idx + 1])[0], "flag": re.findall(r'\((.*?)\)', lines[outputdesc_idx + 2])[0]}
            
            dic[mc_type][kernel_name].append(info_dic)   

    json_object = json.dumps(dic, indent=4)
    with open(json_file, "w") as outfile:
        outfile.write(json_object)

def create_ddiMHA_csv(ddi_file):
    root_path = r"C:\Users\qianshui\OneDrive - Intel Corporation\Desktop\My-File\All_Intel_CCG_GFx_workday\Performance_data\DG2_512_SDXL_perf_logs"

    openf =  open(ddi_file, "r")
    lines = openf.readlines()

    i = -1
    csv_file = os.path.join(root_path,"DDI_MHA.csv")
    csvf = open(csv_file,"w",newline='')
    writer =csv.writer(csvf)

    line = ["mc_type","gemm0_kernel_name", "gemm0_A_size", "gemm0_B_size","gemm0_Output_size",
            "gemm1_kernel_name", "gemm1_A_size", "gemm1_B_size","gemm1_Output_size"]
    writer.writerow(line)
    while i < len(lines)-1:
        i+=1
        if "Passed-Metacommand type : MHA" in lines[i]:
            info_dic =dict()
            mc_type = lines[i].rstrip().split("type :")[-1]
            gemm0_kernel_name = ""
            gemm0_A_size = ""
            gemm0_B_size = ""
            gemm0_Output_size = ""

            gemm1_kernel_name = ""
            gemm1_A_size = ""
            gemm1_B_size = ""
            gemm1_Output_size = ""

            for k in range(50):
                if "m_mhaGemm0Desc.ADesc.Size" in lines[i-k]:
                    print(lines[i-k])
                    gemm0_A_size =  re.findall(r'[[](.*?)[]]', lines[i-k])[0]
                    continue
                if "m_mhaGemm0Desc.BDesc.Size" in lines[i-k]:
                    print(lines[i-k])
                    gemm0_B_size =  re.findall(r'[[](.*?)[]]', lines[i-k])[0]
                    continue
                if "m_mhaGemm0Desc.OutputDesc.Size" in lines[i-k]:
                    print(lines[i-k])
                    gemm0_Output_size =  re.findall(r'[[](.*?)[]]', lines[i-k])[0]
                    continue
                if "MHA Gemm0 Shader Code" in lines[i-k]:
                    # print(lines[i-k].split('=')[1])
                    gemm0_kernel_name =  lines[i-k].split('=')[1][:-1]
                    continue
                
                if "m_mhaGemm1Desc.ADesc.Size" in lines[i-k]:
                    print(lines[i-k])
                    gemm1_A_size =  re.findall(r'[[](.*?)[]]', lines[i-k])[0]
                    continue
                if "m_mhaGemm1Desc.BDesc.Size" in lines[i-k]:
                    print(lines[i-k])
                    gemm1_B_size =  re.findall(r'[[](.*?)[]]', lines[i-k])[0]
                    continue
                if "m_mhaGemm1Desc.OutputDesc.Size" in lines[i-k]:
                    print(lines[i-k])
                    gemm1_Output_size =  re.findall(r'[[](.*?)[]]', lines[i-k])[0]
                    continue
                if "MHA Gemm1 Shader Code" in lines[i-k]:
                    # print(lines[i-k].split('=')[1])
                    gemm1_kernel_name =  lines[i-k].split('=')[1][:-1]
                    continue
            
            writer.writerow([mc_type,gemm0_kernel_name, gemm0_A_size, gemm0_B_size,gemm0_Output_size,
                                gemm1_kernel_name, gemm1_A_size, gemm1_B_size,gemm1_Output_size])

    csvf.close()
    print("{} generated".format(csv_file))
    df = pd.read_csv(csv_file)
    s = pd.pivot_table(df, index=['gemm0_kernel_name', "gemm1_kernel_name"], aggfunc={"gemm0_kernel_name": "count", })
    s.columns=['count']
    ddi_conv_pivot_table = s.sort_values(by='count', ascending=0)
    ddi_conv_pivot_table.to_csv(os.path.join(model_path, "ddi_mha_pivot_table.csv"))
    print(s)



def create_ddiPool_json(ddi_file):
    json_file = os.path.basename(ddi_file).replace(".log","_pool.json")
    dic ={}
    openf =  open(ddi_file, "r")

    lines = openf.readlines()
    #print(len(lines))
    i = -1
    while i < len(lines)-1:
        i+=1
        if "Passed-Metacommand type : Pooling" in lines[i]:
            info_dic ={}
            mc_type = lines[i].rstrip().split("type :")[-1]
            kernel_name_idx = i - 3

            kernel_name = lines[kernel_name_idx].rstrip().split("Kernel:")[-1] 
            kernel_name = "unknown" if kernel_name=="" else kernel_name
            #print(mc_type, kernel_name)
            if mc_type not in dic.keys():
                dic[mc_type] = {}
            if kernel_name not in dic[mc_type].keys():
                dic[mc_type][kernel_name] = []      

           
            layout_list = re.findall(r'N\w+', lines[i+4]) 

            input_shape = []
            for j in range(4):
                input_shape.append(int(lines[i+9+j].rstrip().split("=")[-1],16))
            '''offset is useless'''
            # input_stride = []
            # for j in range(4):
            #     input_stride.append(int(lines[i+13+j].rstrip().split("=")[-1],16))

            outputdesc_idx = i+24
            output_shape = []
            for j in range(4):
                output_shape.append(int(lines[outputdesc_idx + 4 + j].rstrip().split("=")[-1],16))
            '''offset is useless'''
            # output_stride = []
            # for j in range(4):
            #     output_stride.append(int(lines[outputdesc_idx +  8 +j].rstrip().split("=")[-1],16))
            
            available_function = ["AvgPool", "L2Pool", "MaxPool"]  # ref from driver code MetaCommandPoolingDnnl line62
            pooling_function = available_function[int(lines[outputdesc_idx + 19].rstrip().split("=")[-1],16)]
            stride = []
            for j in range(2):
                stride.append(int(lines[outputdesc_idx +  21 +j].rstrip().split("=")[-1],16))
            
            kernel_size = []  #WindowSize
            for j in range(2):
                kernel_size.append(int(lines[outputdesc_idx +  24 +j].rstrip().split("=")[-1],16))

            padding_size = []  #[h_begin, w_begin, h_end, w_end ]
            for j in range(6):
                if j == 2 or j ==5:
                    continue
                padding_size.append(int(lines[outputdesc_idx +  27 +j].rstrip().split("=")[-1],16))    

            '''m_PoolingParams.PoolingType is not kernel size'''
            info_dic["input"] = {"layout": layout_list[0],"shape": input_shape,  "datatype": re.findall(r'\((.*?)\)', lines[i+6])[0], "flag": re.findall(r'\((.*?)\)', lines[i+7])[0]}
            info_dic["filter"] = {"shape": kernel_size, "stride": stride, "padding": padding_size}
            info_dic["pool_function"] = pooling_function
            info_dic["output"] = {"layout": layout_list[1], "shape":output_shape,   "datatype": re.findall(r'\((.*?)\)', lines[outputdesc_idx + 1])[0], "flag": re.findall(r'\((.*?)\)', lines[outputdesc_idx + 2])[0]}
            
            dic[mc_type][kernel_name].append(info_dic)   

    json_object = json.dumps(dic, indent=4)
    with open(json_file, "w") as outfile:
        outfile.write(json_object)

def cvt_json2csv_conv():
    root_path = r"C:\Users\qianshui\OneDrive - Intel Corporation\Desktop\My-File\All_Intel_CCG_GFx_workday\Performance_data\DG2_512_SDXL_perf_logs"
    files = glob.glob(root_path+"\\*_conv.json")

    csv_file = os.path.join(root_path,"DDI_Conv.csv")
    csvf = open(csv_file,"w",newline='')
    writer =csv.writer(csvf)
    line = ["model_name", "kernel_name","input_shape_n","input_shape_c","input_shape_h","input_shape_w", "input_layout", "input_datatype","input_flag","input_padding",\
            "filter_shape_n","filter_shape_c","filter_shape_h","filter_shape_w", "filter_layout", "filter_datatype","filter_flag", "filter_stride_h", "filter_stride_w", "filter_stride_c", "filter_dilation_h", "filter_dilation_w", "filter_dilation_c", "filter_groupcount",
            "output_shape_n","output_shape_c","output_shape_h","output_shape_w", "output_layout", "output_datatype","output_flag", "output_padding", 
            "bias", "direction", "activation" ,"exec_flag"]
    writer.writerow(line)
    for file in files:
        basename = os.path.basename(file)
        df = pd.read_json(file)
        conv =df[df.keys()[0]]
        for key in conv.keys():
            kernel_name = key
            for i in range(len(conv[key])):
                input_shape = conv[key][i]["input"]["shape"]
                input_layout = conv[key][i]["input"]["layout"]
                input_datatype = conv[key][i]["input"]["datatype"]
                input_flag = conv[key][i]["input"]["flag"]
                input_padding = [int(v) for v in  conv[key][i]["input"]["padding"]] 

                filter_shape = conv[key][i]["filter"]["shape"]
                filter_layout = conv[key][i]["filter"]["layout"]
                filter_datatype = conv[key][i]["filter"]["datatype"]
                filter_flag = conv[key][i]["filter"]["flag"]
                filter_stride = conv[key][i]["filter"]["stirde"]
                filter_dilation = conv[key][i]["filter"]["dilation"]
                filter_groupcount = conv[key][i]["filter"]["groupcount"]

                output_shape = conv[key][i]["output"]["shape"]
                output_layout = conv[key][i]["output"]["layout"]
                output_datatype = conv[key][i]["output"]["datatype"]
                output_flag = conv[key][i]["output"]["flag"]
                output_padding = [int(v) for v in conv[key][i]["output"]["padding"]]

                # print(conv[key][i].keys())
                bias = conv[key][i]["bias"]
                direction = conv[key][i]["direction"]
                activation = conv[key][i]["activation"]
                exec_flag = conv[key][i]["exec_flag"]

                writer.writerow([basename, kernel_name, input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_layout, input_datatype, input_flag,input_padding,\
                                filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3], filter_layout, filter_datatype, filter_flag, filter_stride[0], filter_stride[1],filter_stride[2],filter_dilation[0],filter_dilation[1],filter_dilation[2], filter_groupcount, \
                                    output_shape[0],output_shape[1],output_shape[2],output_shape[3], output_layout, output_datatype, output_flag, output_padding,
                                    bias, direction, activation, exec_flag])
    
    csvf.close()

    print("{} generated".format(csv_file))
    df = pd.read_csv(csv_file)
    s = pd.pivot_table(df, index=['kernel_name'], aggfunc={"kernel_name": "count"})
    s.columns=['count']

    ddi_conv_pivot_table = s.sort_values(by='count', ascending=0)
    ddi_conv_pivot_table.to_csv(os.path.join(model_path, "ddi_conv_pivot_table.csv"))

def cvt_json2csv_gemm():
    root_path = r"C:\Users\qianshui\OneDrive - Intel Corporation\Desktop\My-File\All_Intel_CCG_GFx_workday\Performance_data\DG2_512_SDXL_perf_logs"
    files = glob.glob(root_path+"\\*_gemm.json")

    csv_file = os.path.join(root_path,"DDI_GEMM.csv")
    csvf = open(csv_file,"w",newline='')
    writer =csv.writer(csvf)
    line = ["model_name", "kernel_name","inputA_shape_M","inputA_shape_K", "inputA_datatype","inputA_flag","transA",\
            "inputB_shape_K","inputB_shape_N", "inputB_datatype","inputB_flag","transB",\
            "output_shape_M","output_shape_N", "output_datatype","output_flag", "inputC_flag", "alpha","beta" ]
    writer.writerow(line)
    for file in files:
        basename = os.path.basename(file)
        df = pd.read_json(file)
        if len(df.keys()) == 0:
            continue
        #print(basename)
        gemm =df[df.keys()[0]]
        for key in gemm.keys():
            kernel_name = key
            for i in range(len(gemm[key])):
                inputA_shape = gemm[key][i]["inputA"]["shape"]
                inputA_datatype = gemm[key][i]["inputA"]["datatype"]
                inputA_flag = gemm[key][i]["inputA"]["flag"]
                transA = gemm[key][i]["inputA"]["transA"]

                inputB_shape = gemm[key][i]["inputB"]["shape"]
                inputB_datatype = gemm[key][i]["inputB"]["datatype"]
                inputB_flag = gemm[key][i]["inputB"]["flag"]
                transB = gemm[key][i]["inputB"]["transB"]

                output_shape = gemm[key][i]["output"]["shape"]
                output_datatype = gemm[key][i]["output"]["datatype"]
                output_flag = gemm[key][i]["output"]["flag"]

                inputC_flag = gemm[key][i]["inputC"]
                alpha = gemm[key][i]["config"]["alpha"]
                beta = gemm[key][i]["config"]["beta"]
                
                # print(inputA_shape)
                writer.writerow([basename, kernel_name, inputA_shape[0], inputA_shape[1], inputA_datatype, inputA_flag,transA,\
                                inputB_shape[0], inputB_shape[1], inputB_datatype, inputB_flag,transB, \
                                    output_shape[0],output_shape[1],output_datatype, output_flag,\
                                    inputC_flag,alpha, beta])
        
    csvf.close()

def cvt_json2csv_pool():
    root_path = r"C:\Users\qianshui\OneDrive - Intel Corporation\Desktop\My-File\All_Intel_CCG_GFx_workday\Performance_data\DG2_512_SDXL_perf_logs"
    files = glob.glob(root_path+"\\*_pool.json")

    csv_file = os.path.join(root_path,"DDI_Pooling.csv")
    csvf = open(csv_file,"w",newline='')
    writer =csv.writer(csvf)
    # the name is mapping to https://onnx.ai/onnx/operators/onnx__MaxPool.html#maxpool
    line = ["model_name", "kernel_name","pool_function", \
        "input_layout", "input_shape_n","input_shape_c","input_shape_h","input_shape_w", "input_datatype","input_flag",\
        "kernel_shape_h","kernel_shape_w", "stride_h", "stride_w","pads",\
        "output_layout", "output_shape_n","output_shape_c", "output_shape_h","output_shape_w", "output_datatype","output_flag" ]
    writer.writerow(line)
    for file in files:
        basename = os.path.basename(file)
        df = pd.read_json(file)
        if len(df.keys()) == 0:
            continue
        #print(basename)
        pool =df[df.keys()[0]]
        for key in pool.keys():
            kernel_name = key
            for i in range(len(pool[key])):
                pool_function = pool[key][i]["pool_function"]
                input_layout = pool[key][i]["input"]["layout"]
                input_shape = pool[key][i]["input"]["shape"]
                input_datatype = pool[key][i]["input"]["datatype"]
                input_flag = pool[key][i]["input"]["flag"]

                filter_shape =  pool[key][i]["filter"]["shape"]
                filter_stride =  pool[key][i]["filter"]["stride"]
                filter_padding =  pool[key][i]["filter"]["padding"]

                output_layout = pool[key][i]["output"]["layout"]
                output_shape = pool[key][i]["output"]["shape"]
                output_datatype = pool[key][i]["output"]["datatype"]
                output_flag = pool[key][i]["output"]["flag"]

                writer.writerow([basename, kernel_name, pool_function,
                                input_layout, input_shape[0], input_shape[1], input_shape[2], input_shape[3],input_datatype, input_flag, \
                                    filter_shape[0],filter_shape[1], filter_stride[0],filter_stride[1], filter_padding,\
                                    output_layout, output_shape[0], output_shape[1], output_shape[2], output_shape[3],output_datatype, output_flag ])
        
    csvf.close()

def create_extra_ddiMHA_csv(ddi_file):
    root_path = r"C:\Users\qianshui\OneDrive - Intel Corporation\Desktop\My-File\All_Intel_CCG_GFx_workday\Performance_data\DG2_512_SDXL_perf_logs"

    openf =  open(ddi_file, "r")
    lines = openf.readlines()

    i = -1
    csv_file = os.path.join(root_path,"DDI_extra_MHA.csv")
    csvf = open(csv_file,"w",newline='')
    writer =csv.writer(csvf)

    line = ["mc_type","gemm0_kernel_name", "gemm0_A_size", "gemm0_B_size","gemm0_Output_size",
            "gemm1_kernel_name", "gemm1_A_size", "gemm1_B_size","gemm1_Output_size"]
    writer.writerow(line)
    while i < len(lines)-1:
        i+=1
        condition_0 = True if "MHA Gemm1 Shader Code = g_mha_sv_s_qkv_gemm_CM" in lines[i] else False
        condition_1 =  True if 'MHA compiling REFERENCE ocl gemm kernel for QxK case.' in lines[i-1]  else False
        if condition_0 and condition_1:
            info_dic =dict()
            mc_type = "extra_mha"
            gemm0_kernel_name = ""
            gemm0_A_size = ""
            gemm0_B_size = ""
            gemm0_Output_size = ""

            gemm1_kernel_name = ""
            gemm1_A_size = ""
            gemm1_B_size = ""
            gemm1_Output_size = ""

            for k in range(15):
                if "m_mhaGemm0Desc.ADesc.Size" in lines[i-k]:
                    print(lines[i-k])
                    gemm0_A_size =  re.findall(r'[[](.*?)[]]', lines[i-k])[0]
                    continue
                if "m_mhaGemm0Desc.BDesc.Size" in lines[i-k]:
                    print(lines[i-k])
                    gemm0_B_size =  re.findall(r'[[](.*?)[]]', lines[i-k])[0]
                    continue
                if "m_mhaGemm0Desc.OutputDesc.Size" in lines[i-k]:
                    print(lines[i-k])
                    gemm0_Output_size =  re.findall(r'[[](.*?)[]]', lines[i-k])[0]
                    continue
                if "MHA Gemm0 Shader Code" in lines[i-k]:
                    # print(lines[i-k].split('=')[1])
                    gemm0_kernel_name =  lines[i-k].split('=')[1][:-1]
                    continue
                
                if "m_mhaGemm1Desc.ADesc.Size" in lines[i-k]:
                    print(lines[i-k])
                    gemm1_A_size =  re.findall(r'[[](.*?)[]]', lines[i-k])[0]
                    continue
                if "m_mhaGemm1Desc.BDesc.Size" in lines[i-k]:
                    print(lines[i-k])
                    gemm1_B_size =  re.findall(r'[[](.*?)[]]', lines[i-k])[0]
                    continue
                if "m_mhaGemm1Desc.OutputDesc.Size" in lines[i-k]:
                    print(lines[i-k])
                    gemm1_Output_size =  re.findall(r'[[](.*?)[]]', lines[i-k])[0]
                    continue
                if "MHA Gemm1 Shader Code" in lines[i-k]:
                    # print(lines[i-k].split('=')[1])
                    gemm1_kernel_name =  'g_mha_sv_s_qkv_gemm_CM'
                    continue
            
            writer.writerow([mc_type,gemm0_kernel_name, gemm0_A_size, gemm0_B_size,gemm0_Output_size,
                                gemm1_kernel_name, gemm1_A_size, gemm1_B_size,gemm1_Output_size])

    csvf.close()
    print("{} generated".format(csv_file))
    df = pd.read_csv(csv_file)
    s = pd.pivot_table(df, index=['gemm0_kernel_name', "gemm1_kernel_name"], aggfunc={"gemm0_kernel_name": "count", })
    s.columns=['count']
    ddi_conv_pivot_table = s.sort_values(by='count', ascending=0)
    ddi_conv_pivot_table.to_csv(os.path.join(model_path, "ddi_extra_mha_pivot_table.csv"))
    print(s)

if __name__ == "__main__":
    model_path = r'C:\Users\qianshui\OneDrive - Intel Corporation\Desktop\My-File\All_Intel_CCG_GFx_workday\Performance_data\DG2_512_SDXL_perf_logs\DDI_logs'
    log_root = r'C:\Users\qianshui\OneDrive - Intel Corporation\Desktop\My-File\All_Intel_CCG_GFx_workday\Performance_data\DG2_512_SDXL_perf_logs\DDI_logs'
    filename = 'onnxruntime_perf_test0_unet.log'


    for path, dir_list, file_list in os.walk(log_root):
        # print(path)
        if filename in file_list:
            ddi_file_path = os.path.join(path, filename)
            # create_ddiConv_json(ddi_file_path)
            # cvt_json2csv_conv()
            # create_ddiGEMM_json(ddi_file_path)
            # cvt_json2csv_gemm()
            # create_ddiMHA_csv(ddi_file_path)
            create_extra_ddiMHA_csv(ddi_file_path)
            # create_ddiPool_json(ddi_file_path)
            # cvt_json2csv_pool()
            # files = glob.glob("*.json")
            # files = files + glob.glob("*.csv")
