import os
import csv
import collections
import math
import pandas as pd
from broja2pid import BROJA_2PID
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats

import re

import time



Xname_list = ['head_pls','eye_pls']
Y = 2
Z = 3



# # 相当于保留一位小数
# def round_float(x):
#     return round(float(x) * 10, 0)
#     # return round(float(x), 0)


def round_float(x):
    return round(float(x), 1)
    # return round(float(x), 0)

angle_directions = ['Theta','Phi']


vis_names = ['Vis1','Vis2','Vis3']

root_path = "pre_data_seconds_reord"

root_path_files = os.listdir(root_path)

for root_path in root_path_files:

    pattern = re.compile(r'\d+.\d+')
    intval_time = pattern.findall(root_path)
    print(intval_time)



    for X,Xname in enumerate(Xname_list):
            
        for angle_direction in angle_directions:
            for vis_name in vis_names:


                SI_List = []
                UIY_List = []
                UIZ_List = []
                CI_List = []
                Name_list = []


                dir_path = "pre_data_seconds_reord\\" + root_path + "\\walk_head_eye\\"+ angle_direction +'\\'+ vis_name
                files = os.listdir(dir_path)

                for file in files:

                    Name_list.append(file.split('_')[1])
                    path = dir_path +"\\" + file
                    data = {}
                    with open(path, newline='') as csvfile:
                        reader = csv.reader(csvfile)
                        next(reader) 
                        for row in reader:
                            condition = (round_float(row[X]), round_float(row[Y]), round_float(row[Z]))
                            data[condition] = data.get(condition, 0) + 1 
                        
                    total_samples = sum(data.values())
                    prob_distr = {comb: count / total_samples for comb, count in data.items()}

                    print(prob_distr)

                    

                # ECOS parameters 
                    parms = dict()
                    parms['max_iters'] = 100000

                    # print("Starting BROJA_2PID.pid() on AND gate.")
                    try:
                        returndata = BROJA_2PID.pid(prob_distr, cone_solver="ECOS", output=2, **parms)

                        # msg="""
                        #     Shared information: {SI}
                        #     Unique information in Y: {UIY}
                        #     Unique information in Z: {UIZ}
                        #     Synergistic information: {CI}
                        #     Primal feasibility: {Num_err[0]}
                        #     Dual feasibility: {Num_err[1]}
                        #     Duality Gap: {Num_err[2]}
                        #     """
                        # print(msg.format(**returndata))
                    
                    except BROJA_2PID.BROJA_2PID_Exception:
                        print("Cone Programming solver failed to find (near) optimal solution. Please report the input probability density function to abdullah.makkeh@gmail.com")

                    # print("The End")
                

                    SI_List.append(returndata['SI'])
                    UIY_List.append(returndata['UIY'])
                    UIZ_List.append(returndata['UIZ'])
                    CI_List.append(returndata['CI'])

                from statsmodels.stats.stattools import durbin_watson


                dw_statistic = durbin_watson(SI_List)
                print(f"SI Durbin-Watson statistic: {dw_statistic}")

                dw_statistic = durbin_watson(UIY_List)
                print(f"UIY Durbin-Watson statistic: {dw_statistic}")

                dw_statistic = durbin_watson(UIZ_List)
                print(f"UIZ Durbin-Watson statistic: {dw_statistic}")

                dw_statistic = durbin_watson(CI_List)
                print(f"CI Durbin-Watson statistic: {dw_statistic}")

                print("---------------------------------- storing ----------------------------------")
                ############################################################################################
                #                                       data store                                         #
                ############################################################################################
                data_store_Theta = list(zip(SI_List, UIY_List, UIZ_List, CI_List, Name_list))
                csv_path = 'result_seconds_reord\\result_' + intval_time[0] +'s\\walk_head_eye\\' + angle_direction +'\\' + Xname +'\\' + vis_name +'_result.csv'
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                with open(csv_path, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(['SI', 'UIY','UIZ','CI','Name'])
                    csvwriter.writerows(data_store_Theta)

                
