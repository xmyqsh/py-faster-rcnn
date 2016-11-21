# --------------------------------------------------------
# Written by xmyqsh
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import json
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from ucd_eval import ucd_eval
from fast_rcnn.config import cfg

class ucar_driveless(imdb):
    def __init__(self, image_set, bg_or_not, ucd_path=None):
        imdb.__init__(self, 'ucd_' + bg_or_not + '_' + image_set)
        self._bg_or_not = bg_or_not
        self._image_set = image_set
        self._ucd_path = self._get_default_path() if ucd_path is None \
                            else ucd_path
        self._data_path = os.path.join(self._ucd_path, 'training')
        self._classes = ('__background__', # always index 0
                         'vehicle', 'pedestrian', 'cyclist', 'trafficlights')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        self._labels = self._load_ucd_labels()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._ucd_path), \
                'VOCdevkit path does not exist: {}'.format(self._ucd_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._ucd_path + /training/ImageSets/val.txt
        # self._data_path + /ImageSets/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                            self._image_set +
                            ('_with_bg' if self._bg_or_not == 'bg' else '')
                            + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'UCarDriveless')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_ucd_label(index)
                    for index in self.image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _load_ucd_labels(self):
        """
        Load image and bounding boxes info to a json struct from json file
        in the UCar Driveless.
        """
        filename = os.path.join(self._data_path, 'label.idl')
        labels = []
        with open(filename, 'r') as fid:
            labels = [json.loads(line.strip()) for line in fid.readlines()]

        # turn labels from list to dict
        labels_dict = {}
        for label in labels:
            imgId = label.keys()[0]
            rois = label[imgId]
            labels_dict[imgId] = rois

        return labels_dict

    def _load_ucd_label(self, index):
        objs = self.labels[index + self._image_ext]
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x1 = float(obj[0]) - 1
            y1 = float(obj[1]) - 1
            x2 = float(obj[2]) - 1
            y2 = float(obj[3]) - 1
            cls = 4 if obj[-1] == 20 else obj[-1]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # UCarDriveless/results/UCD/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._ucd_path,
            'results',
            'UCD' + ('_with_bg' if self._bg_or_not == 'bg' else ''),
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the UCarDriveless expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _gen_submit_json(self):
        imagenames = [imageidx + self._image_ext for imageidx in self._image_index]
        submit_label = {}
        for imagename in imagenames:
            submit_label[imagename] = []
        for cls in self._classes:
            if cls == '__background__':
                continue

            detpath = self._get_voc_results_file_template().format(cls)
            with open(detpath, 'r') as fid:
                lines = fid.readlines()

            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = [float(x[1]) for x in splitlines]
            BB = [[float(z) for z in x[2:]] for x in splitlines]

            for i, image_id in enumerate(image_ids):
                label_item = BB[i]
                label_item.append(20 if self._class_to_ind[cls] == 4 else self._class_to_ind[cls])
                label_item.append(confidence[i])
                submit_label[image_id + self._image_ext].append(label_item)

        submitpath = os.path.join(
                        self._ucd_path,
                        'results',
                        'UCD',
                        'submit.json')
        with open(submitpath, 'w') as fid:
            encoded_submit_label = json.dumps(submit_label)
            fid.writelines(encoded_submit_label)

    def _do_python_eval(self, output_dir = 'output'):
        imagenames = [imageidx + self._image_ext for imageidx in self._image_index]
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = False# if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        truepostive_label = {}
        for imagename in imagenames:
            truepostive_label[imagename] = []
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            detpath = self._get_voc_results_file_template().format(cls)
            truepostivepath = os.path.join(
                            self._ucd_path,
                            'truepostive_{:s}.json').format(cls)
            rec, prec, ap, truepostive_label_cls = ucd_eval(
                detpath, truepostivepath, self._labels, imagenames, cls, self._class_to_ind[cls], ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
            for imagename in imagenames:
                if len(truepostive_label_cls[imagename]) == 0:
                    continue
                truepostive_label[imagename].extend(truepostive_label_cls[imagename])

        truepostivepath = os.path.join(
                        self._ucd_path,
                        'truepostive.json')
        with open(truepostivepath, 'w') as fid:
            fid.writelines(str(truepostive_label))
            #fid.writelines(json.dumps([truepostive_label]))

        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)



    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        if self._image_set != 'submit':
            self._do_python_eval(output_dir)
        else:
            print('Generating submit.json file')
            self._gen_submit_json()
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.ucar_driveless import ucar_driveless
    d = ucar_driveless('trainval', '')
    res = d.roidb
    from IPython import embed; embed()
