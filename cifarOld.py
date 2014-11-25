#  vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

from numpy import *
import cPickle
from random import randrange

def read_cifar_file(fn):
    fo = open(fn, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_cifar(psz=4):
    #  file strings
    # /home/teddy/Desktop/PyDeSTIN/cifar-10-batches-py
    cifar_dir = '/home/teddy/Desktop/PyDeSTIN/cifar-10-batches-py/'
    # cifar_dir = '/home/syoung22/Data/cifar-10-batches-py/'
    filenames = ['data_batch_1', 'data_batch_2',
                 'data_batch_3', 'data_batch_4',
                 'data_batch_5', 'test_batch']

    #  gather data
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

    #  reshape data into images

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

    #  set dims
    train_data.shape = (50000, 32, 32, 3)
    test_data.shape = (10000, 32, 32, 3)

    #  get random patches
    patches = empty((200000, psz * psz * 3))
    # psz = 4
    for i in range(200000):
        im = randrange(50000)
        a = randrange(32 - psz)
        b = randrange(32 - psz)
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
    eps = 1e-9
    patch_cov = cov(patches, rowvar=0)
    d, e = linalg.eig(patch_cov)
    d = diag(d) + eps
    v = e.dot(linalg.inv(sqrt(d))).dot(e.T)
    patches = patches.dot(v)

    ret = {}
    # ret['train_data'] = train_data
    # ret['test_data'] = test_data
    # ret['train_labels'] = train_labels
    # ret['test_labels'] = test_labels
    ret['patch_mean'] = patch_mean
    ret['patch_std'] = patch_std
    ret['whiten_mat'] = v

    return ret

    # vts = {}
    # vts['images'] = patches
    #  io.savemat('/home/syoung22/Data/cifar.mat',vts)


if __name__ == '__main__':
    load_cifar()