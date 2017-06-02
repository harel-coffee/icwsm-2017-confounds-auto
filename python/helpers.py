import numpy as np
import itertools

from sklearn.preprocessing import LabelEncoder

def absdiff(x,y):
    return np.abs(x-y).tolist()

def absdiff_eps(x,y,eps):
    if eps is not None:
        keep = np.max(y, axis=1) > eps
        x = x[keep]
        y = y[keep]
        y = np.argmax(y, axis=1)
    else:
        r = None
    return np.abs(x-y).tolist()

def make_confounding_data(y, z, bias, size, rand=np.random.RandomState(123456)):
    both_pos = [i for i in range(len(y)) if y[i] == 1 and z[i] == 1]
    both_neg = [i for i in range(len(y)) if y[i] == 0 and z[i] == 0]
    ypos_zneg = [i for i in range(len(y)) if y[i] == 1 and z[i] == 0]
    yneg_zpos = [i for i in range(len(y)) if y[i] == 0 and z[i] == 1]

    for x in [both_pos, both_neg, yneg_zpos, ypos_zneg]:
        rand.shuffle(x)

    # if bias=.9, then 90% of instances where z=1 will also have y=1
    # similarly, 10% of instances where z=1 will have y=0
    zprob = sum(z) / len(z)
    pos_prob = sum(y) / len(y)
    n_zpos = int(zprob * size)
    n_zneg = size - n_zpos
    n_ypos = int(pos_prob * size)
    n_yneg = size - n_ypos

    n_11 = int(bias * n_zpos)
    n_01 = int((1 - bias) * n_zpos)
    n_10 = n_ypos - n_11
    n_00 = n_yneg - n_01

    r = np.array(both_pos[:n_11] + both_neg[:n_00] + ypos_zneg[:n_10] + yneg_zpos[:n_01])
    return r

def get_Xyz(X, users, y_column, z_column):
    le_y = LabelEncoder().fit(users[y_column].unique())
    le_z = LabelEncoder().fit(users[z_column].unique())
    y = le_y.transform(users[y_column])
    z = le_z.transform(users[z_column])
    return X, y, z

def uniform_noise_1d(arr, noise, rand=np.random.RandomState(1191)):
    uniq = set(arr)
    arr = arr.copy()
    flip_bool = rand.uniform(size=len(arr)) <= noise
    for (i, do_flip) in enumerate(flip_bool):
        if do_flip:
            cur = arr[i]
            others = list(uniq - {cur})
            arr[i] = rand.choice(others, size=1)[0]
    return arr