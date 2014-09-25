<<<<<<< HEAD
#  -*- coding: utf-8 -*-
=======
# -*- coding: utf-8 -*-
>>>>>>> ab02f8d2a9e47bf672c5d77c5f60b408df2c9fdb
from numpy import *
import cPickle
from random import randrange
import numpy as np
<<<<<<< HEAD
cifar_dir = '/home/teddy/Desktop/Cifar/'
# cifar_dir = '/home/eskender/Destin/cifar-10-batches-py/'
#  Contains loading cifar batches and
#  feeding input to lower layer nodes


=======
cifar_dir = '/home/teddy/Desktop/PyDeSTIN/cifar-10-batches-py/'
#cifar_dir = '/home/eskender/Destin/cifar-10-batches-py/'
# Contains loading cifar batches and
# feeding input to lower layer nodes
>>>>>>> ab02f8d2a9e47bf672c5d77c5f60b408df2c9fdb
def read_cifar_file(fn):
    fo = open(fn, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def load_cifar(psz=4):
<<<<<<< HEAD
    #  file strings
=======
    # file strings
    # /home/teddy/Desktop/PyDeSTIN/cifar-10-batches-py
    #cifar_dir = '/eskender@ih1:~/Destin/cifar-10-batches-py/'
    #cifar_dir = '/home/syoung22/Data/cifar-10-batches-py/'
>>>>>>> ab02f8d2a9e47bf672c5d77c5f60b408df2c9fdb
    filenames = ['data_batch_1', 'data_batch_2',
                 'data_batch_3', 'data_batch_4',
                 'data_batch_5', 'test_batch']

<<<<<<< HEAD
    #  gather data
=======
    # gather data
>>>>>>> ab02f8d2a9e47bf672c5d77c5f60b408df2c9fdb
    train_data = empty((50000, 3072))
    test_data = empty((10000, 3072))
    train_labels = empty(50000)
    test_labels = empty(10000)

    start = 0
    width = 10000
    for file in filenames:
        dic = read_cifar_file(cifar_dir + file)
        if start < 50000:
            train_data[start:start + width, :] = dic['data']
            train_labels[start:start + width] = array(dic['labels'])
        else:
            test_data[:, :] = dic['data']
            test_labels[:] = array(dic['labels'])

        start += width

<<<<<<< HEAD
    #  reshape data into images
=======
    # reshape data into images
>>>>>>> ab02f8d2a9e47bf672c5d77c5f60b408df2c9fdb

    for x in range(50000):
        image = train_data[x]
        image.shape = (3, 32, 32)
        image2 = copy(image.transpose((1, 2, 0)))
        image2 = reshape(image2, (1, 3072))
        train_data[x] = image2

    for x in range(10000):
        image = test_data[x]
        image.shape = (3, 32, 32)
        image2 = copy(image.transpose((1, 2, 0)))
        image2 = reshape(image2, (1, 3072))
        test_data[x] = image2

<<<<<<< HEAD
    #  set dims
    train_data.shape = (50000, 32, 32, 3)
    test_data.shape = (10000, 32, 32, 3)

    #  get random patches
    patches = empty((200000, psz * psz * 3))
    # psz = 4
=======

    # set dims
    train_data.shape = (50000, 32, 32, 3)
    test_data.shape = (10000, 32, 32, 3)

    # get random patches
    patches = empty((200000, psz * psz * 3))
    #psz = 4
>>>>>>> ab02f8d2a9e47bf672c5d77c5f60b408df2c9fdb
    for i in range(200000):
        im = randrange(50000)
        a = randrange(32 - psz)
        b = randrange(32 - psz)
<<<<<<< HEAD
        patch = reshape(
            train_data[im, a:a + psz, b:b + psz, :], (1, psz * psz * 3))
        patches[i] = patch

    #  get statistics
    patch_mean = mean(patches, axis=0)
    patch_std = std(patches, axis=0)

    #  zero mean and unit variance
    patches = patches - patch_mean
    patches = patches / patch_std

    #  whitening stuff using notation from:
    #  http://web.eecs.utk.edu/~itamar/Papers/ICMLA2012_Derek.pdf
=======
        patch = reshape(train_data[im, a:a + psz, b:b + psz, :], (1, psz * psz * 3))
        patches[i] = patch

    # get statistics
    patch_mean = mean(patches, axis=0)
    patch_std = std(patches, axis=0)

    # zero mean and unit variance
    patches = patches - patch_mean
    patches = patches / patch_std

    # whitening stuff using notation from:
    # http://web.eecs.utk.edu/~itamar/Papers/ICMLA2012_Derek.pdf
>>>>>>> ab02f8d2a9e47bf672c5d77c5f60b408df2c9fdb
    eps = 1e-9
    patch_cov = cov(patches, rowvar=0)
    d, e = linalg.eig(patch_cov)
    d = diag(d) + eps
    v = e.dot(linalg.inv(sqrt(d))).dot(e.T)
    patches = patches.dot(v)

    ret = {}
<<<<<<< HEAD
    # ret['train_data'] = train_data
    # ret['test_data'] = test_data
    # ret['train_labels'] = train_labels
    # ret['test_labels'] = test_labels
=======
    #ret['train_data'] = train_data
    #ret['test_data'] = test_data
    #ret['train_labels'] = train_labels
    #ret['test_labels'] = test_labels
>>>>>>> ab02f8d2a9e47bf672c5d77c5f60b408df2c9fdb
    ret['patch_mean'] = patch_mean
    ret['patch_std'] = patch_std
    ret['whiten_mat'] = v

    return ret


def loadCifar(batchNum):
<<<<<<< HEAD
    #  For training_batches specify numbers 1 to 5
    #  for the test set pass 6
=======
    # For training_batches specify numbers 1 to 5
    # for the test set pass 6
>>>>>>> ab02f8d2a9e47bf672c5d77c5f60b408df2c9fdb
    if batchNum <= 5:
        FileName = cifar_dir + '/data_batch_' + str(batchNum)
        FID = open(FileName, 'rb')
        dict = cPickle.load(FID)
        FID.close()
        return dict['data'], dict['labels']
    elif batchNum == 6:
        FileName = cifar_dir + '/test_batch'
        FID = open(FileName, 'rb')
        dict = cPickle.load(FID)
        FID.close()
        return dict['data'], dict['labels']
    else:  # here we will get the whole 50,000x3072 dataset
        I = 0
        FileName = cifar_dir + '/data_batch_' + str(I + 1)
        FID = open(FileName, 'rb')
        dict = cPickle.load(FID)
        FID.close()
        data = dict['data']
        labels = dict['labels']
        for I in range(1, 5):
            FileName = cifar_dir + '/data_batch_' + str(I + 1)
            FID = open(FileName, 'rb')
            dict = cPickle.load(FID)
            FID.close()
            data = np.concatenate((data, dict['data']), axis=0)
            labels = np.concatenate((labels, dict['labels']), axis=0)
        return data, labels


def returnNodeInput(Input, Position, Ratio, Mode, ImageType):
    if Mode == 'Adjacent':  # Non overlapping or Adjacent Patches
        PatchWidth = Ratio
        PatchHeight = Ratio
        if ImageType == 'Color':
            PatchDepth = 3
        else:
            PatchDepth = 1
        Patch = Input[Position[0]:Position[0] + PatchWidth, Position[1]:Position[1] + PatchHeight].reshape(1,
                                                                                                           PatchWidth * PatchWidth * PatchDepth)
    else:  # TODO Overlapping Patch could be fed to a node
        print('Overlapping Patches Are Not Implemented Yet')
        patch = np.array([])
    return Patch
