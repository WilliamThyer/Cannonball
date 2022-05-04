import decode_eeg
import pandas as pd
import numpy as np
import scipy.io as io

experiment_name = 'C01'
data_dir = f'./data/{experiment_name}'
exp = decode_eeg.Experiment(experiment_name, data_dir, test=False)

# match_list = [('05','06'),('07','08'),('09', '10'),('11', '12')]
match_list = [('13', '14')]

# find file indices for matched subjects
match_list_idx = []
for m in match_list:
    for i in range(exp.nsub):    

        if exp.info_files[i].name[4:6] == m[0]:
            a = i
        if exp.info_files[i].name[4:6] == m[1]:
            b = i

    match_list_idx.append((a, b))

# load and concatenate data from behavior, eeg, and trial labels
for m,i in enumerate(match_list_idx): 
    x1, y1 = exp.load_eeg(i[0])
    x2, y2 = exp.load_eeg(i[1])
    idx1 = exp.load_artifact_idx(i[0])
    idx2 = exp.load_artifact_idx(i[1])
    beh1 = exp.load_behavior(i[0], remove_artifact_trials=False)
    beh2 = exp.load_behavior(i[1], remove_artifact_trials=False)

    x = np.concatenate((x1, x2), 0)
    x = np.moveaxis(x, 0, 2) # roll bc saving with savemat changes shape for some reason...
    y = np.concatenate((y1, y2), 0)
    idx = np.concatenate((idx1, idx2), 0)
    beh1_df = pd.DataFrame(beh1)
    beh2_df = pd.DataFrame(beh2)
    sub_num = beh1_df.loc[0,'Subject']
    beh2_df.loc[:,'Subject'] = sub_num
    beh = pd.concat((beh1_df, beh2_df))

    # save
    io.savemat(f'data/C01/C01_{match_list[m][0]}_xdata.mat',{'xdata':x})
    io.savemat(f'data/C01/C01_{match_list[m][0]}_ydata.mat',{'ydata':y})
    io.savemat(f'data/C01/C01_{match_list[m][0]}_artifact_idx.mat',{'artifact_idx':idx})
    beh.to_csv(f'data/C01/C01_{match_list[m][0]}_behavior.csv')