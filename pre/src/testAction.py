from Action import *
from general import *
a = Action("../../Data/OAD/skeleton/S9.txt", 25)
a.read_label("../../Data/OAD/label/L9.txt")
n = np.array(a.norm_point_seq)
s = np.array(a.savitzky_point_seq)
t1 = np.array(a.taylor_1_point_seq)
t2 = np.array(a.taylor_2_point_seq)
regression = np.array(a.regression_seq)

index = a.frame_seq.index(176)
index_ = a.frame_seq.index(144)
print(n[:,index])
print(s[:,index])
print(t1[:,index])
print(t2[:,index])
print(index)
print(a.label_seq[index])
print(a.label_seq[index_])
print(regression[index])


