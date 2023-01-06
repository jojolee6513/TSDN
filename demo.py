# -*- coding: utf-8 -*-
from Network.ops import *
from Network.models import *
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input,Flatten,Dense,Lambda
from Network.graph import GraphConvolution
from keras.optimizers import Adam
import time
from Network.utils import *
import os
from DataReader.generate_dataset import generate_dataset_train,get_idx_train_val, creat_train
from Network.para import *
import tifffile as tiff
import scipy.io as sio
import keras
from Network.utils import *





def test(network):
    ###############################################新代码#################################################
    # if network == 'at':
    #     model = spat_pred()
    #     model.load_weights(model_at)
    #     Xh = np.load('./file/train_Xh.npy')
    #     Xl = np.load('./file/train_Xl.npy')
    #     pred = model.predict([Xh, Xl])
    #     np.save('pred_at.npy', pred)
    # if network == 'ec':
    #     model = spec_pred()
    #     model.load_weights(model_ec)
    #     Xh = np.load('./file/train_Xh.npy')
    #     pred = model.predict([Xh[:, r, r, :, np.newaxis]])
    #     np.save('pred_ec.npy', pred)
    # ###############################################新代码#################################################
    if network == 'test':
        model = new_fusion()
        model.load_weights(modelname)
        model.load_weights("model/MUUFL_fusion_weights.h5")
        # [Xl, Xh] = make_fusionTest()
        # [Xl, Xh] = make_CNNTest()
        generate_dataset_train(flag='cnn_te')
        Xh = np.load('./file/hsi.npy')
        Xl = np.load('./file/lidar.npy')
        pred = model.predict([Xh, Xh[:, r, r, :, np.newaxis], Xl])

        np.save('pred_cnn.npy', pred)
        OA, AA, Kappa = cvt_map(pred, show=False)
        print('OA: {:.2f}%  AA: {:.2f}%  Kappa: {:.4f}'.format(OA, AA, Kappa))
    if network == 'test2':
        model = new_fusion()
        model.load_weights(modelname_f)
        # [Xl, Xh] = make_fusionTest()
        # [Xl, Xh] = make_CNNTest()
        generate_dataset_train(flag='cnn_te')
        Xh = np.load('./Image/file/hsi.npy')
        Xl = np.load('./Image/file/lidar.npy')
        pred = model.predict([Xh, Xh[:, r, r, :, np.newaxis], Xl])

        np.save('pred_f.npy', pred)
        OA, AA, Kappa = cvt_map(pred, show=False)
        print('OA: {:.2f}%  AA: {:.2f}%  Kappa: {:.4f}'.format(OA, AA, Kappa))



def main():

    # generate_dataset_train(flag='cnn_tr', validation=False, rate=0.86)
    # generate_dataset_train(flag='cnn_tr', validation=True, rate=0.86)
    # # #
    # # test each branch
    # # atmodel = spat_pred()
    # # train_at(atmodel)
    # # # # # test('at')
    # # ecmodel = spec_pred()
    # # train_ec(ecmodel)
    # # # # test('ec')
    # model = new_fusion()
    # model.summary()
    # fffusion(model)
    # # start = time.time()

    # test('test')
    # # # test('train')
    # # print('elapsed time:{:.2f}s'.format(time.time() - start))
    # # #
    # gcn()


    # pesudo = sio.loadmat('out.mat')['out']
    #
    # true = sio.loadmat(os.path.join(PATH, gth_train))['train']
    # # pesudo = np.zeros_like(true)
    # new_train = true + pesudo
    # np.save('new_train.npy', new_train)
    # creat_train()
    # model = new_fusion()
    # train(model)
    test('test2')


def train(model):
    Xl_train = np.load('./Image/file/new_train_Xl.npy')
    Xh_train = np.load('./Image/file/new_train_Xh.npy')
    Y_train = keras.utils.np_utils.to_categorical(np.load('./Image/file/new_train_Y.npy'))
    Xl_val = np.load('./Image/file/val_Xl.npy')
    Xh_val = np.load('./Image/file/val_Xh.npy')
    Y_val = keras.utils.np_utils.to_categorical(np.load('./Image/file/val_Y.npy'))
    model_ckt = ModelCheckpoint(filepath=modelname_f, verbose=1, monitor='val_loss', save_best_only=True)
    model.fit([Xh_train, Xh_train[:, r, r, :, np.newaxis], Xl_train], Y_train, batch_size=64, epochs=epochs,
              callbacks=[model_ckt], validation_data=([Xh_val, Xh_val[:, r, r, :, np.newaxis], Xl_val], Y_val))
    scores = model.evaluate([Xh_val, Xh_val[:, r, r, :, np.newaxis], Xl_val], Y_val, batch_size=64)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    model.save(modelname_f)

def fffusion(model):
    # 训练并保存模型
    Xl_train = np.load('./file/train_Xl.npy')
    Xh_train = np.load('./file/train_Xh.npy')
    Y_train = keras.utils.np_utils.to_categorical(np.load('./file/train_Y.npy'))
    Xl_val = np.load('./file/val_Xl.npy')
    Xh_val = np.load('./file/val_Xh.npy')
    Y_val = keras.utils.np_utils.to_categorical(np.load('./file/val_Y.npy'))
    model_ckt = ModelCheckpoint(filepath=modelname, verbose=1, monitor='val_loss', save_best_only=True)
    model.fit([Xh_train, Xh_train[:, r, r, :, np.newaxis], Xl_train], Y_train, batch_size=BATCH_SIZE, epochs=epochs,
              callbacks=[model_ckt], validation_data=([Xh_val, Xh_val[:, r, r, :, np.newaxis], Xl_val], Y_val))
    scores = model.evaluate([Xh_val, Xh_val[:, r, r, :, np.newaxis], Xl_val], Y_val, batch_size=BATCH_SIZE)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    model.save(modelname)

def train_at(model):
    Xl_train = np.load('./file/train_Xl.npy')
    Xh_train = np.load('./file/train_Xh.npy')
    Y_train = keras.utils.np_utils.to_categorical(np.load('./file/train_Y.npy'))
    Xl_val = np.load('./file/val_Xl.npy')
    Xh_val = np.load('./file/val_Xh.npy')
    Y_val = keras.utils.np_utils.to_categorical(np.load('./file/val_Y.npy'))
    model_ckt = ModelCheckpoint(filepath=model_at, verbose=1, monitor='val_acc', save_best_only=True)
    model.fit([Xh_train,  Xl_train], Y_train, batch_size=BATCH_SIZE, epochs=epochs,
              callbacks=[model_ckt], validation_data=([Xh_val, Xl_val], Y_val))
    model.save(model_at)

def train_ec(model):
    Xh_train = np.load('./file/train_Xh.npy')
    Y_train = keras.utils.np_utils.to_categorical(np.load('./file/train_Y.npy'))
    Xh_val = np.load('./file/val_Xh.npy')
    Y_val = keras.utils.np_utils.to_categorical(np.load('./file/val_Y.npy'))
    model_ckt = ModelCheckpoint(filepath=model_ec, verbose=1, monitor='val_acc', save_best_only=True)
    model.fit([Xh_train[:, r, r, :, np.newaxis]], Y_train, batch_size=BATCH_SIZE, epochs=epochs,
              callbacks=[model_ckt], validation_data=([Xh_val[:, r, r, :, np.newaxis]], Y_val))
    model.save(model_ec)


def gcn(NB_EPOCH = 2500, classNum = NUM_CLASS, weight_dis = 0.005):


    # Get data

    X, A, y, y_train, y_test, idx_train, idx_test, train_mask = generate_dataset_train(flag = 'gcn',num_scale=num_scale)
    idx_train, idx_val = get_idx_train_val(y_train[idx_train], np.array(idx_train), classNum, 0.1)  # 获取idx_train、idx_val
    y_val = np.zeros((y_train.shape[0], classNum))
    y_val[idx_val] = y_train[idx_val]
    y_train[idx_val] = np.zeros((idx_val.shape[0], classNum))
    train_mask[idx_val] = False
    # Normalize X
    # X /= X.sum(1).reshape(-1, 1)
    X = sample_wise_normalization(X)


    SYM_NORM = True
    MAX_DEGREE = 2
    G = []
    graph = []
    support = MAX_DEGREE + 1
    for idx_scale in range(num_scale):
        L = normalized_laplacian(A[idx_scale], SYM_NORM)
        L_scaled = rescale_laplacian(L)
        T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
        graph += T_k
        G += [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]

    X_in = Input(shape=(X.shape[1],))

    Y = []
    for idx_scale in range(num_scale):  #

        H = GraphConvolution(25, support, activation='relu')([X_in] + G[idx_scale * support:(idx_scale + 1) * support])
        H = Dropout(0.5)(H)  # 此行代码可以注释掉或者修改括号里的值，对于CE_loss 建议注释掉，对于DCE_loss建议使用0.1或0.5
        Y += [GraphConvolution(y_train.shape[1], support, activation='relu')([H] + G[idx_scale * support:(idx_scale + 1) * support])]

    scale = 1  # '0':3*3  '1':5*5  '2':7*7
    H = GraphConvolution(32, support, activation='relu')([X_in] + G[scale * support:(scale + 1) * support])
    output = GraphConvolution(y_train.shape[1], support, activation='softmax')([H] + G[scale * support:(scale + 1) * support])
    # get gth
    y_train = np.argmax(y_train, 1)
    y_test = np.argmax(y_test, 1)
    y_val = np.argmax(y_val, 1)

    # Compile model

    model = Model(inputs=[X_in]+G, outputs=[output] + [output])
    # model = Model(inputs=[X_in]+G, outputs=[output,output])
    model.summary()
    model.compile(loss=['sparse_categorical_crossentropy', dis_loss], loss_weights=[1.0, weight_dis],
                  optimizer=Adam(lr=0.01))

    # Fit
    train_t = time.time()
    for epoch in range(1, NB_EPOCH + 1):
        # Log wall-clock time
        t = time.time()
        # sample_weight = logsig((np.ones(y_train.shape[0], dtype='float32') - 1 + epoch-1 - NB_EPOCH / 2) / NB_EPOCH * 10)
        # Single training iteration (we mask nodes without labels for loss calculation)
        model.fit([X] + graph, [y_train, y_train], sample_weight=[train_mask, np.ones((y_train.shape[0]))],
                  batch_size=A[0].shape[0], epochs=1, shuffle=False, verbose=0)

        # Predict on full dataset
        preds, _ = model.predict([X] + graph, batch_size=A[0].shape[0])

        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                       [idx_train, idx_val], weight_dis)

        print("Epoch: {:04d}".format(epoch),
              "train_loss= {:.4f}".format(train_val_loss[0]),
              "train_acc= {:.4f}".format(train_val_acc[0]),
              "val_loss= {:.4f}".format(train_val_loss[1]),
              "val_acc= {:.4f}".format(train_val_acc[1]),
              "time= {:.4f}".format(time.time() - t))

    print("Training time =", str(time.time() - train_t))
    # Testing
    test_t = time.time()
    test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test], weight_dis)
    print("Testing time =", str(time.time() - test_t))

    print("Test set results:",
          "loss= {:.4f}".format(test_loss[0]),
          "accuracy= {:.4f}".format(test_acc[0]))
    np.save('pred_gcn_all.npy', preds)
    # preds = preds[2832:15028,...]
    preds = preds[2832:15029,...]
    np.save('pred_gcn.npy', preds)



if __name__ == '__main__':
    main()


