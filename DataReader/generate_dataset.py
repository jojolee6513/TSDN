from Network.utils import *
from scipy import sparse
from Network.utils import normalize
from Network.models import *
import keras

def generate_dataset_train(flag='cnn_tr', validation=False, rate=0.9, num_scale = 1):

    hsi = read_mat(PATH,HSIName,'Hdata')
    x_all = read_mat(PATH, HSIName, 'Hdata')
    lidar = read_mat(PATH,LiDARName,'lidar')
    y = read_mat(PATH, gth, 'label')
    y_tr = read_mat(PATH,gth_train,'train')
    y_te = read_mat(PATH, gth_test,'test')


    hsi = np.pad(hsi, ((r, r), (r, r), (0, 0)), 'symmetric')
    x_all = np.pad(x_all, ((r, r), (r, r), (0, 0)), 'symmetric')
    lidar = np.pad(lidar, ((r, r), (r, r), (0, 0)), 'symmetric')
    y = np.pad(y, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    y_tr = np.pad(y_tr, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    y_te = np.pad(y_te, ((r, r), (r, r)), 'constant', constant_values=(0, 0))


    lidar = sample_wise_standardization(lidar)
    hsi = sample_wise_standardization(hsi)



    posx, posy =np.where(y != 0)
    pixels = y.shape[0] * y.shape[1]
    x_all = x_all.reshape(pixels, hchn)
    x_all = x_all.astype(float)
    x_all = sample_wise_standardization(x_all)

    y_all = y.reshape(pixels, 1)
    # 生成全部数据索引
    idx = np.where(y_all>0)
    num_nodes = idx[0].shape[0]
    x_all = x_all[idx[0]]
    y_all = y_all[idx[0]]
    y_all -= 1
    # 创建 train_idx val_idx text_idx
    y_tr = y_tr.reshape(pixels, 1)[idx]
    y_te = y_te.reshape(pixels, 1)[idx]
    # 生成训练集和测试集数据索引
    idx_train = np.squeeze(np.array(np.where(y_tr > 0)), axis=0)
    tr_samples = len(idx_train)
    idx_test = np.squeeze(np.array(np.where(y_te > 0)), axis=0)
    np.random.shuffle(idx_train)
    np.random.shuffle(idx_test)
    sio.savemat('./Image/MUUFL/idx_train.mat', {"idx_train": idx_train})
    sio.savemat('./Image/MUUFL/idx_test.mat', {"idx_test": idx_test})
    idx_test = sio.loadmat('./Image/MUUFL/idx_test.mat')['idx_test']
    idx_test = np.squeeze(idx_test)
    idx_train = sio.loadmat('./Image/MUUFL/idx_train.mat')['idx_train']
    idx_train = np.squeeze(idx_train)
    tr_samples = len(idx_train)
    idx_all = np.squeeze(np.hstack((idx_train, idx_test)))
    sio.savemat('./Image/MUUFL/MUUFL_sort_shuffle_idx_all.mat', {"idx_all": idx_all})




    if flag == 'cnn_tr':
        posx_tr, posy_tr = posx[idx_train], posy[idx_train]

        Xh = []
        Xl = []
        Y = []
        if not validation:
            posx_tr = posx_tr[:int(rate * tr_samples)]
            posy_tr = posy_tr[:int(rate * tr_samples)]
            idx_train = idx_train[:int(rate * tr_samples)]
        else:
            posx_tr = posx_tr[int(rate * tr_samples):]
            posy_tr = posy_tr[int(rate * tr_samples):]
            idx_train = idx_train[int(rate * tr_samples):]
        for i in range(len(posy_tr)):
            tmph = hsi[posx_tr[i] - r: posx_tr[i] + r + 1, posy_tr[i] - r: posy_tr[i] + r + 1, :]
            tmpl = lidar[posx_tr[i] - r: posx_tr[i] + r + 1, posy_tr[i] - r: posy_tr[i] + r + 1]
            tmpy = y[posx_tr[i], posy_tr[i]] - 1
            # 翻转，加噪声，旋转90度
            Xh.append(tmph)
            Xh.append(np.flip(tmph, axis=0))
            noise = np.random.normal(0.0, 0.01, size=tmph.shape)
            Xh.append(np.flip(tmph + noise, axis=1))
            k = np.random.randint(4)
            Xh.append(np.rot90(tmph, k=k))
            # 翻转，加噪声，旋转90度
            Xl.append(tmpl)
            Xl.append(np.flip(tmpl, axis=0))
            noise = np.random.normal(0.0, 0.03, size=tmpl.shape)
            Xl.append(np.flip(tmpl + noise, axis=1))
            Xl.append(np.rot90(tmpl, k=k))
            # gth不变
            Y.append(tmpy)
            Y.append(tmpy)
            Y.append(tmpy)
            Y.append(tmpy)


        index = np.random.permutation(len(Xh))
        Xh = np.asarray(Xh, dtype=np.float32)
        Xl = np.asarray(Xl, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int8)
        Xh = Xh[index, ...]
        Y = Y[index]
        Xh = np.asarray(Xh, dtype=np.float32)
        Xl = np.asarray(Xl, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.int8)

        if not validation:
            np.save(os.path.join(SAVA_PATH, 'train_Xh.npy'), Xh)
            np.save(os.path.join(SAVA_PATH, 'train_Xl.npy'), Xl)
            np.save(os.path.join(SAVA_PATH, 'train_Y.npy'), Y)
            np.save(os.path.join(SAVA_PATH, 'train_index.npy'), index)
            print('train hsi data shape:{},train lidar data shape:{}'.format(Xh.shape, Xl.shape))
        else:
            np.save(os.path.join(SAVA_PATH, 'val_Xh.npy'), Xh)
            np.save(os.path.join(SAVA_PATH, 'val_Xl.npy'), Xl)
            np.save(os.path.join(SAVA_PATH, 'val_Y.npy'), Y)
            np.save(os.path.join(SAVA_PATH, 'val_index.npy'), index)

    if flag == 'cnn_te':
        Xh = []
        Xl = []

        for i in range(len(idx_test)):
            tmph = hsi[posx[idx_test[i]] - r: posx[idx_test[i]] + r + 1,
                   posy[idx_test[i]] - r: posy[idx_test[i]] + r + 1, :]
            tmpl = lidar[posx[idx_test[i]] - r: posx[idx_test[i]] + r + 1,
                   posy[idx_test[i]] - r: posy[idx_test[i]] + r + 1]
            Xh.append(tmph)
            Xl.append(tmpl)

        Xh = np.asarray(Xh, dtype=np.float32)
        Xl = np.asarray(Xl, dtype=np.float32)

        np.save(os.path.join(SAVA_PATH, 'hsi.npy'), Xh)
        np.save(os.path.join(SAVA_PATH, 'lidar.npy'), Xl)
        #index = [posx[idx_test] - r, posy[idx_test] - r]
        np.save(os.path.join(SAVA_PATH, 'index.npy'), [posx[idx_test] - r, posy[idx_test] - r])
        #
        Xh = []
        Xl = []

        for i in range(len(idx_train)):
            tmph = hsi[posx[idx_train[i]] - r: posx[idx_train[i]] + r + 1,
                   posy[idx_train[i]] - r: posy[idx_train[i]] + r + 1, :]
            tmpl = lidar[posx[idx_train[i]] - r: posx[idx_train[i]] + r + 1,
                   posy[idx_train[i]] - r: posy[idx_train[i]] + r + 1]
            Xh.append(tmph)
            Xl.append(tmpl)

        Xh = np.asarray(Xh, dtype=np.float32)
        Xl = np.asarray(Xl, dtype=np.float32)

        np.save(os.path.join(SAVA_PATH, 'hsi_test.npy'), Xh)
        np.save(os.path.join(SAVA_PATH, 'lidar_test.npy'), Xl)
        np.save(os.path.join(SAVA_PATH, 'index_test.npy'), [posx[idx_train] - r, posy[idx_train] - r])


    if flag == 'gcn':

        path = r'./Image/'
        dataset = 'MUUFL'

        posx, posy = posx[idx_all],posy[idx_all]
        # posx, posy = posx[idx_test],posy[idx_test]
        np.save(os.path.join(SAVA_PATH, 'index2.npy'), [posx-r,posy-r])
        # idx_all = sio.loadmat('./Image/HU2012/Houston_sort_shuffle_idx_all.mat')['idx_all']
        y_train = np.zeros((num_nodes, NUM_CLASS))
        y_train[0:idx_train.shape[0]] = keras.utils.to_categorical(y_all[idx_train], NUM_CLASS)
        y_test = np.zeros((num_nodes, NUM_CLASS))
        y_test[idx_train.shape[0]:] = keras.utils.to_categorical(y_all[idx_test], NUM_CLASS)
        train_mask = np.zeros((num_nodes))

        train_mask[0:idx_train.shape[0]] = 1
        x_all = x_all[idx_all]
        y_all = y_all[idx_all]

        # 创建邻接矩阵 依据KNN创建
        # adj = []
        # K = 10
        # A = np.zeros((num_nodes, num_nodes))
        # for i in range(num_nodes):
        #     for j in range(i):
        #         dist = eucliDist2(x_all[j], x_all[i])
        #         if dist < 25:
        #             A[i][j] = dist
        #             A[j][i] = dist
        # for scl in range(1, num_scale + 1):
        #     for i in range(num_nodes):
        #         A[i][np.argpartition(A[i], -K * scl)[:-K * scl]] = 0
        #     A = A + A.T * (A.T > A) - A * (A.T > A)
        #     adj.append(sparse.csr_matrix(np.exp(-0.1 * A) * (A > 0)))  # 采用行优先的方式压缩矩阵

        # # 创建邻接矩阵 依据邻域scale创建
        # A = np.zeros((num_nodes, num_nodes))
        # s = np.arange(data_shape[0] * data_shape[1])
        # idx_orig = s[idx]
        # idx_sh = idx_orig[idx_all]
        # adj = []
        # for scl in range(1, num_scale+ 1):
        #     scale = 2 * scl + 1
        #     for i in range(num_nodes):
        #         idx_x_y = [(int)(idx_sh[i] / data_shape[1]), idx_sh[i] % data_shape[1]]
        #         x = np.arange(scale * scale) - int((scale ** 2 - 1) / 2)
        #         x = np.round(x / scale).reshape((scale, scale))
        #         idx_tmp = np.array([x.flatten(), x.T.flatten()])
        #         idx_tmp = np.delete(idx_tmp, ((scale ** 2 - 1) / 2), axis=1)
        #           # 创造转移向量
        #         idx_tmp = np.array([(idx_x_y[0] + idx_tmp[0]), idx_x_y[1] + idx_tmp[1]])
        #         idx_mask = np.array(
        #             (idx_tmp >= 0)[0] * (idx_tmp >= 0)[1] * (idx_tmp[0] < data_shape[0]) * (idx_tmp[1] < data_shape[1]))
        #         idx_tmp = np.squeeze(idx_tmp[:, np.where(idx_mask)], axis=1)
        #         idx_tmp = idx_tmp[0] * data_shape[1] + idx_tmp[1]
        #         for x in idx_tmp:
        #             tp = np.where(idx_sh == x)
        #             if tp[0].shape[0] == 1:
        #                 A[i][tp] = eucliDist2(x_all[i], x_all[tp])
        #     A = A + A.T * (A.T > A) - A * (A.T > A)
        #     adj.append(sparse.csr_matrix(np.exp(-0.1 * A) * (A > 0)))  # 采用行优先的方式压缩矩阵
        # sio.savemat(path+dataset+'/Houston_sort_shuffle_A.mat', {"adj": adj})
        # # 上述生成第一次完成后，可使用下面的代码进行导入，不用再次计算，不使用如下代码则注释掉即可


        A1 = sio.loadmat(path + dataset + '/TrentoS_sort_shuffle_A.mat')['adj']
        adj = [((A1[0, x] > 0) * 1).tocsr() for x in range(num_scale)]

        return np.mat(x_all), adj, y_all, y_train, y_test, range(idx_train.shape[0]), range(idx_train.shape[0],num_nodes), train_mask.astype(bool)
        #return np.mat(x_all), adj, y_all, y_train, y_test, idx_train, idx_test, train_mask.astype(bool)

def get_idx_train_val(labels, idx_train, num_classes, rate):
    val_idx = list()
    mask = np.ones((idx_train.shape[-1]), dtype=np.bool)
    for i in range(num_classes):
        idx_c = (np.where(np.argmax(labels, 1) == i))[0]
        num_c = idx_c.shape[0]
        val_idx.append(idx_c[-round(num_c * rate):])
    idx_val = np.concatenate(val_idx)
    mask[idx_val] = False
    return idx_train[(mask.tolist())], idx_val

#
#
#
def creat_train(val = False,val_rate = 0.85):
    # 读取原始输入
    hsi = sio.loadmat(os.path.join(PATH, HSIName))['Hdata']
    lidar = sio.loadmat(os.path.join(PATH, LiDARName))['lidar']
    gth = np.load('new_train.npy')
    # hsi = read_mat('./','hsi8r3.mat', 'out_hsi')
    # lidar = read_mat('./','lidar8r3.mat', 'out_lidar')
    # gth = tiff.imread(os.path.join(PATH, 'train_new.tif'))

    hsi = np.pad(hsi, ((r, r), (r, r), (0, 0)), 'symmetric')
    if len(lidar.shape) == 2:
        lidar = np.pad(lidar, ((r, r), (r, r)), 'symmetric')
    if len(lidar.shape) == 3:
        lidar = np.pad(lidar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gth = np.pad(gth, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    '''大约9：1划分训练集和验证集,可调参数'''
    lidar = sample_wise_standardization(lidar)
    hsi = sample_wise_standardization(hsi)
    Xh = []
    Xl = []
    Y = []
    for c in range(1, NUM_CLASS + 1):
        idx, idy = np.where(gth == c)
        np.random.seed(2021)
        ID = np.random.permutation(len(idx))
        idx = idx[ID]
        idy = idy[ID]
        # if not val:
        #     idx = idx[:int(val_rate * len(idx))]
        #     idy = idy[:int(val_rate * len(idy))]
        # else:
        #     idx = idx[int(val_rate * len(idx)):]
        #     idy = idy[int(val_rate * len(idy)):]
        for i in range(len(idx)):
            tmph = hsi[idx[i] - r: idx[i] + r + 1, idy[i] - r: idy[i] + r + 1, :]
            tmpl = lidar[idx[i] - r: idx[i] + r + 1, idy[i] - r: idy[i] + r + 1]
            tmpy = gth[idx[i], idy[i]] - 1
            # 翻转，加噪声，旋转90度
            Xh.append(tmph)
            Xh.append(np.flip(tmph, axis=0))
            noise = np.random.normal(0.0, 0.01, size=tmph.shape)
            Xh.append(np.flip(tmph + noise, axis=1))
            k = np.random.randint(4)
            Xh.append(np.rot90(tmph, k=k))
            # 翻转，加噪声，旋转90度
            Xl.append(tmpl)
            Xl.append(np.flip(tmpl, axis=0))
            noise = np.random.normal(0.0, 0.03, size=tmpl.shape)
            Xl.append(np.flip(tmpl + noise, axis=1))
            Xl.append(np.rot90(tmpl, k=k))
            # gth不变
            Y.append(tmpy)
            Y.append(tmpy)
            Y.append(tmpy)
            Y.append(tmpy)

    index = np.random.permutation(len(Xh))
    Xh = np.asarray(Xh, dtype=np.float32)
    Xl = np.asarray(Xl, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int8)

    Xh = Xh[index, ...]
    Y = Y[index]
    np.save(os.path.join(SAVA_PATH, 'new_train_Xh.npy'), Xh)
    np.save(os.path.join(SAVA_PATH, 'new_train_Xl.npy'), Xl)
    np.save(os.path.join(SAVA_PATH, 'new_train_Y.npy'), Y)
    # if not val:
    #     np.save(os.path.join(SAVA_PATH, 'new_train_Xh.npy'), Xh)
    #     np.save(os.path.join(SAVA_PATH, 'new_train_Xl.npy'), Xl)
    #     np.save(os.path.join(SAVA_PATH, 'new_train_Y.npy'), Y)
    # else:
    #     np.save(os.path.join(SAVA_PATH, 'new_val_Xh.npy'), Xh)
    #     np.save(os.path.join(SAVA_PATH, 'new_val_Xl.npy'), Xl)
    #     np.save(os.path.join(SAVA_PATH, 'new_val_Y.npy'), Y)





