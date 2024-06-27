import os
import csv
import re
import pandas as pd
def create_ddiMHA_csv( root_path,ddi_file):

    openf =  open(os.path.join(root_path,ddi_file), "r")
    lines = openf.readlines()

    i = -1
    csv_file = os.path.join(root_path, basename.replace(".log","_mha.csv"))
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
                    gemm0_A_size =  re.findall(r'[[](.*?)[]]', lines[i-k])[0]
                    continue
                if "m_mhaGemm0Desc.BDesc.Size" in lines[i-k]:
                    gemm0_B_size =  re.findall(r'[[](.*?)[]]', lines[i-k])[0]
                    continue
                if "m_mhaGemm0Desc.OutputDesc.Size" in lines[i-k]:
                    gemm0_Output_size =  re.findall(r'[[](.*?)[]]', lines[i-k])[0]
                    continue
                if "MHA Gemm0 Shader Code" in lines[i-k] or "BLOB Gemm Kernel" in lines[i-k]:
                    if "MHA Gemm0 Shader Code" in lines[i-k]:
                        gemm0_kernel_name =  lines[i-k].split('=')[1][:-1]
                    else:
                        gemm0_kernel_name = lines[i-k].split("BLOB Gemm Kernel")[-1].strip().split(".cpp")[0]
                    continue
                
                if "m_mhaGemm1Desc.ADesc.Size" in lines[i-k]:
                    gemm1_A_size =  re.findall(r'[[](.*?)[]]', lines[i-k])[0]
                    continue
                if "m_mhaGemm1Desc.BDesc.Size" in lines[i-k]:
                    gemm1_B_size =  re.findall(r'[[](.*?)[]]', lines[i-k])[0]
                    continue
                if "m_mhaGemm1Desc.OutputDesc.Size" in lines[i-k]:
                    gemm1_Output_size =  re.findall(r'[[](.*?)[]]', lines[i-k])[0]
                    continue
                if "MHA Gemm1 Shader Code" in lines[i-k] or "BLOB Gemm Kernel" in lines[i-k]:
                    if "MHA Gemm1 Shader Code" in lines[i-k]:
                        gemm1_kernel_name =  lines[i-k].split('=')[1][:-1]
                    else:
                        gemm1_kernel_name = lines[i-k].split("BLOB Gemm Kernel")[-1].strip().split(".cpp")[0]
                    continue
            
            writer.writerow([mc_type,gemm0_kernel_name, gemm0_A_size, gemm0_B_size,gemm0_Output_size,
                                gemm1_kernel_name, gemm1_A_size, gemm1_B_size,gemm1_Output_size])

    csvf.close()
    print("{} generated".format(csv_file))
    df = pd.read_csv(csv_file)
    s = pd.pivot_table(df, index=['gemm0_kernel_name', "gemm1_kernel_name"], aggfunc={"gemm0_kernel_name": "count", })
    s.columns=['count']
    ddi_conv_pivot_table = s.sort_values(by='count', ascending=0)
    ddi_conv_pivot_table.to_csv(os.path.join(root_path, "ddi_mha_pivot_table.csv"))
    print(s)

root_path = r"C:\Users\GAME\Documents\Project\helpWindow\onednn_lnl\llm"
basename = "llama_val_16749.log"
create_ddiMHA_csv(root_path, basename)

