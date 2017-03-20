from ICMEW_Action import Action
import numpy as np

ACTION = Action("../../Data/ICMEW/skeleton/train/0190-R.txt"
                , "../../Data/ICMEW/label/train/0190-R.txt")
RAW = np.asarray(ACTION.action_seq)
SMOOTH = ACTION.savitzky_seq
LABEL = ACTION.label_seq
T1 = ACTION.taylor_seq[0]
T2 = ACTION.taylor_seq[1]
REGRESSION = ACTION.regression_seq

INDEX = 250
#print(RAW[INDEX][:75])
#print(SMOOTH[INDEX][:75])
#print(T1[INDEX][:75])
#print(T2[INDEX][:75])
#print(LABEL[INDEX])
#print(REGRESSION[INDEX])
#print(ACTION.TWO)

print(LABEL.shape)
print(SMOOTH.shape)
