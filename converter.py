#!/usr/bin/python
# -*- coding: utf8 -*-

# This script converts .seq files into .jpg files, .vbb files into .pkl files
# from Caltech Pedestrian Dataset
# Based on Python 2.7
# Author: Peng Zhang
# E-mail: hizhangp@gmail.com
# Caltech Pedestrian Dataset:
# http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/

import struct
import os
import cPickle
import time
from scipy.io import loadmat
from collections import defaultdict


def read_seq(path):
    def read_header(ifile):
        feed = ifile.read(4)
        norpix = ifile.read(24)
        version = struct.unpack('@i', ifile.read(4))
        length = struct.unpack('@i', ifile.read(4))
        assert(length != 1024)
        descr = ifile.read(512)
        params = [struct.unpack('@i', ifile.read(4))[0] for i in range(9)]
        fps = struct.unpack('@d', ifile.read(8))
        ifile.read(432)
        image_ext = {100: 'raw', 102: 'jpg', 201: 'jpg', 1: 'png', 2: 'png'}
        return {'w': params[0], 'h': params[1], 'bdepth': params[2],
                'ext': image_ext[params[5]], 'format': params[5],
                'size': params[4], 'true_size': params[8],
                'num_frames': params[6]}

    assert path[-3:] == 'seq', path
    ifile = open(path, 'rb')
    params = read_header(ifile)
    bytes = open(path, 'rb').read()

    imgs = []
    extra = 8
    s = 1024
    for i in range(params['num_frames']):
        tmp = struct.unpack_from('@I', bytes[s:s + 4])[0]
        I = bytes[s + 4:s + tmp]
        s += tmp + extra
        if i == 0:
            val = struct.unpack_from('@B', bytes[s:s + 1])[0]
            if val != 0:
                s -= 4
            else:
                extra += 8
                s += 8
        imgs.append(I)

    return imgs


def read_vbb(path):
    assert path[-3:] == 'vbb'
    vbb = loadmat(path)
    nFrame = int(vbb['A'][0][0][0][0][0])
    objLists = vbb['A'][0][0][1][0]
    maxObj = int(vbb['A'][0][0][2][0][0])
    objInit = vbb['A'][0][0][3][0]
    objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
    objStr = vbb['A'][0][0][5][0]
    objEnd = vbb['A'][0][0][6][0]
    objHide = vbb['A'][0][0][7][0]
    altered = int(vbb['A'][0][0][8][0][0])
    log = vbb['A'][0][0][9][0]
    logLen = int(vbb['A'][0][0][10][0][0])

    data = {}
    data['nFrame'] = nFrame
    data['maxObj'] = maxObj
    data['log'] = log.tolist()
    data['logLen'] = logLen
    data['altered'] = altered
    data['frames'] = defaultdict(list)

    for frame_id, obj in enumerate(objLists):
        if len(obj) > 0:
            for id, pos, occl, lock, posv in zip(obj['id'][0],
                                                 obj['pos'][0],
                                                 obj['occl'][0],
                                                 obj['lock'][0],
                                                 obj['posv'][0]):
                keys = obj.dtype.names
                id = int(id[0][0]) - 1  # MATLAB is 1-origin
                p = pos[0].tolist()
                pos = [p[0] - 1, p[1] - 1, p[2], p[3]]  # MATLAB is 1-origin
                occl = int(occl[0][0])
                lock = int(lock[0][0])
                posv = posv[0].tolist()

                datum = dict(zip(keys, [id, pos, occl, lock, posv]))
                datum['lbl'] = str(objLbl[datum['id']])
                # MATLAB is 1-origin
                datum['str'] = int(objStr[datum['id']]) - 1
                # MATLAB is 1-origin
                datum['end'] = int(objEnd[datum['id']]) - 1
                datum['hide'] = int(objHide[datum['id']])
                datum['init'] = int(objInit[datum['id']])

                data['frames'][frame_id].append(datum)

    return data


if __name__ == '__main__':
    # directory to store data
    dir_path = './'
    # phase can be 'train_', 'test_' or 'val_'
    phase = ''
    # num ranges from 0~11
    num = [0, 11]

    time_flag = time.time()
    img_save_path = os.path.join(dir_path, phase + 'images')
    anno_save_path = os.path.join(dir_path, phase + 'annotations.pkl')
    if os.path.exists(img_save_path):
        raise KeyError('Already exists : {}'.format(img_save_path))
    else:
        os.mkdir(img_save_path)
    print 'Images will be saved to {}'.format(img_save_path)
    print 'Annotations will be saved to {}'.format(anno_save_path)

    #  convert .seq file into .jpg
    for i in range(num[0], num[1]):
        img_set_path = os.path.join(dir_path, 'set{:02}'.format(i))
        assert os.path.exists(
            img_set_path), 'Not exists: '.format(img_set_path)
        print 'Extracting images from set{:02} ...'.format(i)
        for j in sorted(os.listdir(img_set_path)):
            imgs_path = os.path.join(img_set_path, j)
            imgs = read_seq(imgs_path)
            for ix, img in enumerate(imgs):
                img_name = 'img{:02}{}{:04}.jpg'.format(i, j[2:4], ix)
                img_path = os.path.join(img_save_path, img_name)
                open(img_path, 'wb+').write(img)

    print 'Images have been saved.'

    # convert .vbb file into .pkl
    # example: anno['00']['00']['frames'][0][0]['pos']
    anno = defaultdict(dict)
    for i in range(num[0], num[1]):
        anno['{:02}'.format(i)] = defaultdict(dict)
        anno_set_path = os.path.join(dir_path, 'annotations',
                                     'set{:02}'.format(i))
        assert os.path.exists(anno_set_path), \
            'Not exists: '.format(anno_set_path)
        print 'Extracting annotations from set{:02} ...'.format(i)
        for j in sorted(os.listdir(anno_set_path)):
            anno_path = os.path.join(anno_set_path, j)
            anno['{:02}'.format(i)][j[2:4]] = read_vbb(anno_path)

    with open(anno_save_path, 'wb') as f:
        cPickle.dump(anno, f)

    print 'Annotations have been saved.'

    print 'Done, time spends: {}s'.format(int(time.time() - time_flag))
