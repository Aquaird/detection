import numpy as np

def f1(c_pred, r_pred, labels, frame_index, length):

    threshold = 0.5

    c_pred = c_pred[0:length]
    r_pred = r_pred[0:length]
    labels = labels[0:length]
    frame_index = frame_index[0:length]

    label = 0
    ps = []
    perior = [0.0,0.0, 0]

    mlabel = 0
    mps = []
    mperior = [0.0,0.0, 0]

    top_b_c = 0.0
    true_b = -1
    true_e = -1
    top_e_c = 0.0

    for i in range(0, length):
        if labels[i] == 0:
            if label != 0:
                perior[1] = i
                perior[2] = label
                ps.append(perior)
                perior = [0.0,0.0, 0]
                label = 0
        else:
            if label == 0:
                perior[0] = i + 10
                label = labels[i]
                perior[2] = label

        if c_pred[i] == 0:
            if mlabel != 0:
                mperior[0] = true_b
                mperior[1] = true_e
                mperior[2] = mlabel
                mps.append(mperior)
                mperior = [0.0, 0.0, 0]
                true_b = -1
                true_e = -1
                top_b_c = 0
                top_e_c = 0
                mlabel = 0
        else:
            if mlabel == 0:
                mperior[0] = i
                mlabel = c_pred[i]
                top_b_c = r_pred[i][0]
                true_b = i
                top_e_c = r_pred[i][1]
                true_e = i
            else:
                if r_pred[i][0] >= top_b_c:
                    top_b_c = r_pred[i][0]
                    true_b = i
                if r_pred[i][1] >= top_e_c:
                    top_e_c = r_pred[i][1]
                    true_e = i
    print("===========================================")
    print(mps)
    print(ps)

    f1_set = []
    catch_list = []
    for i in range(0,11):
        f1_set.append([0,0,0])

    for i in mps:
        if(i[0] > i[1]):
            continue
        else:
            f1_set[i[2]][1] += 1
            for j in ps:
                alpha_value = alpha(i[0], i[1], j[0], j[1])
                if i[2] == j[2] and alpha_value >= threshold:
                    print(i, j, alpha_value)
                    f1_set[i[2]][0] += 1
                    f1_set[i[2]][1] -= 1
                    catch_list.append(j)
                    break
    for i in catch_list:
        ps.remove(i)

    print(ps)
    for i in ps:
        f1_set[i[2]][2] += 1
    print(f1_set)
    print("===========================================")
    return np.array(f1_set)

def alpha(s,e,ts,te):
    if e<ts or s>te:
        return 0
    else:
        sorted_list = sorted([s,e,ts,te])
        return (sorted_list[2] - sorted_list[1] +1)/(sorted_list[3] - sorted_list[0] +1)
