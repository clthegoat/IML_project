import numpy as np  
from itertools import chain

triplets = np.loadtxt('/Users/zimengjiang/Downloads/task4_handout/train_triplets.txt')
print(triplets.shape)

# jpg_ids = np.arange(1,10000+1)
# num_train = 9000
# num_val = 1000

jpg_ids = np.array(list(set(triplets.reshape(triplets.size))))
num_train = int(0.9*jpg_ids.size)
num_val = jpg_ids.size - num_train
print("num of jpgs in train_all triplets: ", jpg_ids.size)

idx = np.arange(0,jpg_ids.size)
train_jpg_idx = np.random.choice(idx, num_train, replace=False)
train_jpg = jpg_ids[train_jpg_idx]
val_jpg = np.delete(jpg_ids, train_jpg_idx)
print(train_jpg.size)
print(val_jpg.size)

train_triplets_row = []

# find triplets containing training image
for i in range(num_train):
    mask_row, mask_col = np.where(triplets == (train_jpg[i]))
    train_triplets_row.append(list(set(mask_row)))
train_triplets_row = np.array(list(set(list(chain.from_iterable(train_triplets_row)))))
train_triplets_tmp = triplets[train_triplets_row,:]
val_triplets_tmp = np.delete(triplets, train_triplets_row, axis=0)

# # delete training triplet which contains val image
train_del_row = []
for i in range(num_val):
    mask_row, mask_col = np.where(train_triplets_tmp == (val_jpg[i]))
    train_del_row.append(list(set(mask_row)))
train_del_row = np.array(list(set(list(chain.from_iterable(train_del_row)))))
train_triplets_final = np.delete(train_triplets_tmp, train_del_row, axis=0)
print("train_triplets: ", train_triplets_final.shape)

# # delete val triplet which contains train image
val_del_row = []
for i in range(num_train):
    mask_row, mask_col = np.where(val_triplets_tmp == (train_jpg[i]))
    val_del_row.append(list(set(mask_row)))
val_del_row = np.array(list(set(list(chain.from_iterable(val_del_row)))))
val_triplets_final = np.delete(val_triplets_tmp, val_del_row, axis=0)
print("val_triplets: ", val_triplets_final.shape)

np.save("train_triplets",train_triplets_final)
np.save("val_triplets",val_triplets_final)

train = np.load("train_triplets.npy")
val = np.load("val_triplets.npy")
train_set = set(train.reshape(train.size))
val_set = set(val.reshape(val.size))
intersec = train_set.intersection(val_set)
print("intersection: ", len(intersec))




