# coding: utf-8

import struct, os, cPickle
from scipy.io import loadmat
from collections import defaultdict


def read_seq(path, save):
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
        image_ext = {100:'raw', 102:'jpg', 201:'jpg', 1:'png', 2:'png'}
        return {'w':params[0], 'h':params[1], 'bdepth':params[2],
                'ext':image_ext[params[5]], 'format':params[5],
                'size':params[4], 'true_size':params[8],
                'num_frames':params[6]}

    ifile = open(path, 'rb')
    params = read_header(ifile)
    bytes = open(path, 'rb').read()

    extra = 8
    s = 1024
    for i in range(params['num_frames']):
        tmp = struct.unpack_from('@I', bytes[s:s+4])[0]
        I = bytes[s+4:s+tmp]
        s += tmp + extra
        if i == 0:
            val = struct.unpack_from('@B', bytes[s:s+1])[0]
            if val != 0:
                s -= 4
            else:
                extra += 8
                s += 8

        filename = save + 'img' + path[33:35] + path[38:40] + '%.4d.jpg' % i
        open(filename, 'wb+').write(I)


def read_vbb(path):
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
            for id, pos, occl, lock, posv in zip(obj['id'][0], obj['pos'][0],
                obj['occl'][0], obj['lock'][0], obj['posv'][0]):
                keys = obj.dtype.names
                id = int(id[0][0]) - 1  # MATLAB is 1-origin
                pos = pos[0].tolist()
                occl = int(occl[0][0])
                lock = int(lock[0][0])
                posv = posv[0].tolist()

                datum = dict(zip(keys, [id, pos, occl, lock, posv]))
                datum['lbl'] = str(objLbl[datum['id']])
                datum['str'] = int(objStr[datum['id']])
                datum['end'] = int(objEnd[datum['id']])
                datum['hide'] = int(objHide[datum['id']])
                datum['init'] = int(objInit[datum['id']])

                data['frames'][frame_id].append(datum)

    return data


dir_path = './'
anno = defaultdict(dict)

#  convert .seq file into .jpg
for i in range(11):
    anno['%.2d' % i] = defaultdict(dict)
    for j in os.listdir(dir_path + 'set%.2d' % i):
        data_path = dir_path + 'set%.2d/' % i + j
        if i < 6:
            read_seq(data_path, dir_path + 'train/')
        else:
            read_seq(data_path, dir_path + 'test/')


# convert .vbb file into .pkl
for i in range(11):
    anno['%.2d' % i] = defaultdict(dict)
    for k in os.listdir(dir_path + 'annotations/set%.2d' % i):
        anno_path = dir_path + 'annotations/set%.2d/' % i + k
        anno['%.2d' % i][k[2:4]] = read_vbb(anno_path)

with open(dir_path + 'annotations.pkl', 'wb') as f:
    cPickle.dump(anno, f)
