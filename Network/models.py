# -*- coding: utf-8 -*-
from __future__ import print_function, division
import keras as K
import keras.layers as L
from keras.layers import Input,Dropout
from keras.layers import Reshape, multiply, GlobalAveragePooling2D, GlobalAveragePooling1D, GlobalMaxPool1D
from Network.graph import GraphConvolution
from Network.para import *



def gcn_branch(X_in,G, gth_train,support =3):
    Y = []
    for idx_scale in range(num_scale):  #

        H = GraphConvolution(25, support, activation='relu')([X_in] + G[idx_scale * support:(idx_scale + 1) * support])
        H = Dropout(0.5)(H)  # 此行代码可以注释掉或者修改括号里的值，对于CE_loss 建议注释掉，对于DCE_loss建议使用0.1或0.5
        Y += [GraphConvolution(gth_train.shape[1], support, activation='relu')(
            [H] + G[idx_scale * support:(idx_scale + 1) * support])]

    scale = 1  # '0':3*3  '1':5*5  '2':7*7
    H = GraphConvolution(32, support, activation='relu')([X_in] + G[scale * support:(scale + 1) * support])
    output = GraphConvolution(gth_train.shape[1], support, activation='softmax')([H] + G[scale * support:(scale + 1) * support])

    return output



def hsi_spat_feature(input_tensor):
    #(?,6,6,200)
    filters=[32,64,100,200,256]
    conv0_spat=L.Conv2D(filters[2],(3,3),padding='same',activation='relu')(input_tensor)
    conv0_spat=L.BatchNormalization(axis=-1)(conv0_spat)
    conv1_spat=L.Conv2D(filters[2],(3,3),padding='same',activation='relu')(conv0_spat)
    conv1_spat=L.BatchNormalization(axis=-1)(conv1_spat)
    conv2_spat=L.Conv2D(filters[3],(1,1),padding='same',activation='relu')(conv1_spat)
    conv2_spat=L.BatchNormalization(axis=-1)(conv2_spat)
    conv3_spat=L.Conv2D(filters[3],(1,1),padding='same',activation='relu')(conv2_spat)
    conv3_spat=L.BatchNormalization(axis=-1)(conv3_spat)
    pool1=L.MaxPool2D(pool_size=(2,2),padding='same')(conv3_spat)
    feature = pool1
    return feature

def lidar_spat_feature(input_tensor):
    #(?,6,6,128)
    filters = [16, 32, 64, 96, 128, 192, 256, 512]
    conv0 = L.Conv2D(filters[2], (3, 3), padding='same')(input_tensor)
    conv0 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)
    conv0 = cascade_block(conv0, filters[2])
    conv0 = L.MaxPool2D(pool_size=(2, 2), padding='same')(conv0)
    conv1 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)
    conv1 = cascade_block(conv1, nb_filter=filters[4])
    feature = conv1
    return feature

def hsi_spec_feature(input_tensor):
    # (?,hchn,1)
    filters = [8, 16, 32, 64, 128, 256]

    conv0 = L.Conv1D(filters[3], 54, padding='valid')(input_tensor)
    conv0 = L.BatchNormalization(axis=-1)(conv0)
    conv0 = L.advanced_activations.LeakyReLU(alpha=0.20)(conv0)
    conv0 = L.Conv1D(filters[2], 1, padding='valid')(conv0)
    conv0 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)
    conv0 = L.MaxPool1D(pool_size=2, padding='valid')(conv0)
    feature0 = conv0

    # conv1 = L.Conv1D(filters[3], 73, padding='valid')(input_tensor)
    # conv1 = L.BatchNormalization(axis=-1)(conv1)
    # conv1 = L.advanced_activations.LeakyReLU(alpha=0.20)(conv1)
    # conv1 = L.Conv1D(filters[3], 1, padding='valid')(conv1)
    # conv1 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv1)
    # conv1 = L.MaxPool1D(pool_size=2, padding='valid')(conv1)
    # feature1 = conv1
    #
    # conv2 = L.Conv1D(filters[3], 109, padding='valid')(input_tensor)
    # conv2 = L.BatchNormalization(axis=-1)(conv2)
    # conv2 = L.advanced_activations.LeakyReLU(alpha=0.20)(conv2)
    # conv2 = L.Conv1D(filters[4], 1, padding='valid')(conv2)
    # conv2 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv2)
    # conv2 = L.MaxPool1D(pool_size=2, padding='valid')(conv2)
    # feature2 = conv2
    #
    # conv3 = L.Conv1D(filters[3], 127, padding='valid')(input_tensor)
    # conv3 = L.BatchNormalization(axis=-1)(conv3)
    # conv3 = L.advanced_activations.LeakyReLU(alpha=0.20)(conv3)
    # conv3 = L.Conv1D(filters[5], 1, padding='valid')(conv3)
    # conv3 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv3)
    # conv3 = L.MaxPool1D(pool_size=2, padding='valid')(conv3)
    # feature3 = conv3

    # return feature0, feature1, feature2, feature3
    return feature0
    # return feature1, feature2, feature3
''' casecade模块实现'''
def cascade_block(input, nb_filter, kernel_size=3):

    conv1_1 = L.Conv2D(nb_filter * 2, (kernel_size, kernel_size), padding='same')(input)  # nb_filters*2
    conv1_1 = L.BatchNormalization(axis=-1)(conv1_1)
    conv1_1 = L.Activation('relu')(conv1_1)
    conv1_2 = L.Conv2D(nb_filter, (1, 1), padding='same')(conv1_1)  # nb_filters
    conv1_2 = L.BatchNormalization(axis=-1)(conv1_2)
    relu1 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv1_2)
    x = L.Conv2D(nb_filter * 2, (1, 1), use_bias=False, padding='same')(input)
    conv2_1 = L.Conv2D(nb_filter * 2, (kernel_size, kernel_size), padding='same')(relu1)  # nb_filters*2
    conv2_1 = L.Add()([x, conv2_1])
    conv2_1 = L.BatchNormalization(axis=-1)(conv2_1)
    conv2_1 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv2_1)
    conv2_2 = L.Conv2D(nb_filter, (3, 3), padding='same')(conv2_1)  # nb_filters
    conv2_2 = L.BatchNormalization(axis=-1)(conv2_2)
    conv2_2 = L.Add()([conv1_2, conv2_2])
    relu2 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv2_2)

    return relu2

def joint_spat(hsi_in,lidar_in):


    feature1 = hsi_spat_feature(hsi_in)
    feature2 = lidar_spat_feature(lidar_in)
    feature = L.concatenate([feature1, feature2])

    # att_spatinput = (feature.shape.dims[1].value,feature.shape.dims[2].value,feature.shape.dims[3].value)
    # model = resnet_v1.resnet_v1(input_shape=att_spatinput, depth=20, num_classes=NUM_CLASS,attention_module=Attention)
    # output = model(feature)
    output = feature
    fea1 = featuremap_attention(output)
    output = multiply([fea1,output])
    fea2 = spat_attention(output)
    output = multiply([fea2,output])


    return output

def spec(spec_in):

    filters = [8, 16, 32, 64, 128, 256]

    # feature0, feature1, feature2, feature3 = hsi_spec_feature(spec_in)
    feature0 = hsi_spec_feature(spec_in)
    output0 = channel_attention(feature0)
    output0 = multiply([feature0,output0])

    return output0


###############################################新代码#################################################
def spat_pred():
    ksize = 2 * r + 1
    hsi_in = L.Input((ksize, ksize, hchn))
    lidar_in = L.Input((ksize, ksize, lchn))

    feature_spat = joint_spat(hsi_in,lidar_in)
    spat_out = L.Flatten()(feature_spat)
    out = L.Dropout(0.5)(spat_out)
    logits = L.Dense(NUM_CLASS, activation='softmax')(out)
    model = K.models.Model([hsi_in,lidar_in],[logits])
    adam = K.optimizers.Adam(lr=0.00005)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy', metrics=['acc'])
    return model

def spec_pred():
    spec_in = L.Input((hchn, 1))
    feature_spec = spec(spec_in)
    spec_out = L.Flatten()(feature_spec)
    out = L.Dropout(0.5)(spec_out)
    logits = L.Dense(NUM_CLASS, activation='softmax')(out)
    model = K.models.Model([spec_in],[logits])
    adam = K.optimizers.Adam(lr=0.00005)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy', metrics=['acc'])
    return model


###############################################新代码#################################################

def new_fusion():
    ksize = 2 * r + 1
    hsi_in = L.Input((ksize, ksize, hchn))
    lidar_in = L.Input((ksize, ksize, lchn))
    spec_in = L.Input((hchn, 1))

    feature_spat = joint_spat(hsi_in,lidar_in)
    feature_spec = spec(spec_in)

    spat_out = L.Flatten()(feature_spat)
    spec_out = L.Flatten()(feature_spec)

    out = L.concatenate([spat_out,spec_out])
    out = L.Dropout(0.5)(out)
    logits = L.Dense(NUM_CLASS, activation='softmax')(out)


    model = K.models.Model([hsi_in,spec_in,lidar_in], logits)
    adam = K.optimizers.Adam(lr=0.00005)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy', metrics=['acc'])

    return model


def featuremap_attention(input_feature):

    channels = input_feature.shape.dims[3].value
    out = GlobalAveragePooling2D()(input_feature)
    out = Reshape((1, channels))(out)
    out = L.Conv1D(channels, 1, activation='relu', padding='valid')(out)
    out = L.Conv1D(channels, 1, activation='relu', padding='valid')(out)
    out = L.Conv1D(channels, 1, activation='sigmoid')(out)
    out = GlobalAveragePooling1D()(out)

    return out

def spat_attention(input_feature):

    out = L.Conv2D(1, (4,4), padding='same',activation='relu')(input_feature)
    # out = L.Conv2D(1, (2,2), padding='same',activation='relu')(out)
    out = L.Conv2D(1, (1,1), padding='same',activation='sigmoid')(out)
    # out = K.activations.sigmoid(out)
    # out = K.backend.sigmoid(out)

    return out



def channel_attention(input_feature):

    channels = input_feature.shape.dims[2].value
    # out = GlobalAveragePooling1D()(input_feature)
    out = GlobalMaxPool1D()(input_feature)
    out = Reshape((1, channels))(out)
    out = L.Conv1D(channels, 1, activation='relu', padding='valid')(out)
    out = L.Conv1D(channels, 1, activation='relu', padding='valid')(out)
    out = L.Conv1D(channels, 1, activation='sigmoid')(out)
    out = GlobalAveragePooling1D()(out)

    return out

