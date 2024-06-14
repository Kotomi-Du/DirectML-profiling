import json
import csv
import re
import pandas as pd
import os

def create_ddiPOOL_csv(root_path, basename, generate_cmd_flag = False):
    ddi_file = os.path.join(root_path, basename)
    csv_file = os.path.join(root_path, basename.replace(".log","_pool.csv"))
    openf =  open(ddi_file, "r")

    csvf = open(csv_file,"w",newline='')
    writer =csv.writer(csvf)
    # the name is mapping to https://onnx.ai/onnx/operators/onnx__MaxPool.html#maxpool
    line = ["model_name", "kernel_name","pool_function", \
        "input_layout", "input_shape_n","input_shape_c","input_shape_h","input_shape_w", "input_datatype","input_flag",\
        "kernel_shape_h","kernel_shape_w", "stride_h", "stride_w","pads",\
        "output_layout", "output_shape_n","output_shape_c", "output_shape_h","output_shape_w", "output_datatype","output_flag" ]
    writer.writerow(line)

    lines = openf.readlines()
    #print(len(lines))
    i = -1
    while i < len(lines)-1:
        i+=1
        if "Passed-Metacommand type : Pooling" in lines[i]:
            info_dic ={}
            mc_type = lines[i].rstrip().split("type :")[-1]
            kernel_name_idx = i + 61
            print(lines[kernel_name_idx])
            if "Kernel:" in lines[kernel_name_idx]:
                kernel_name = lines[kernel_name_idx].rstrip().split("Kernel:")[-1] 
                kernel_name = "unknown" if kernel_name=="" else kernel_name
            else:
                kernel_name = "no MC used"
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
            filter_stride = []
            for j in range(2):
                filter_stride.append(int(lines[outputdesc_idx +  21 +j].rstrip().split("=")[-1],16))
            
            filter_shape = []  #WindowSize
            for j in range(2):
                filter_shape.append(int(lines[outputdesc_idx +  24 +j].rstrip().split("=")[-1],16))

            filter_padding = []  #[h_begin, w_begin, h_end, w_end ]
            for j in range(6):
                if j == 2 or j ==5:
                    continue
                filter_padding.append(int(lines[outputdesc_idx +  27 +j].rstrip().split("=")[-1],16))    

            '''m_PoolingParams.PoolingType is not kernel size'''

            input_layout =layout_list[0]
            output_layout = layout_list[1]
            input_datatype =re.findall(r'\((.*?)\)', lines[i+6])[0]
            input_flag =  re.findall(r'\((.*?)\)', lines[i+7])[0]
            output_datatype =   re.findall(r'\((.*?)\)', lines[outputdesc_idx + 1])[0]
            output_flag = re.findall(r'\((.*?)\)', lines[outputdesc_idx + 2])[0]

            writer.writerow([basename, kernel_name, pooling_function,
                             input_layout, input_shape[0], input_shape[1], input_shape[2], input_shape[3],input_datatype, input_flag, \
                                filter_shape[0],filter_shape[1], filter_stride[0],filter_stride[1], filter_padding,\
                                output_layout, output_shape[0], output_shape[1], output_shape[2], output_shape[3],output_datatype, output_flag ])
    csvf.close()

root_path = r"C:\Users\GAME\Documents\Project\helpWindow\onednn_lnl\post_pv"
basename = "alex_newOS.log"

create_ddiPOOL_csv(root_path, basename)