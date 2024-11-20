import numpy as np

np.bool = np.bool_

import mxnet as mx
import cv2
from mxnet.gluon.model_zoo import vision

from .config import config


def mround(x):
    x_ = x.copy()
    idx = (x - np.floor(x)) >= 0.5
    x_[idx] = np.floor(x[idx]) + 1
    idx = ~idx
    x_[idx] = np.floor(x[idx])
    return x_

class Feature:
    def init_size(self, img_sample_sz, cell_size=None):
        if cell_size is not None:
            max_cell_size = max(cell_size)
            new_img_sample_sz = (1 + 2 * mround(img_sample_sz / ( 2 * max_cell_size))) * max_cell_size
            feature_sz_choices = np.array([(new_img_sample_sz.reshape(-1, 1) + np.arange(0, max_cell_size).reshape(1, -1)) // x for x in cell_size])
            num_odd_dimensions = np.sum((feature_sz_choices % 2) == 1, axis=(0,1))
            best_choice = np.argmax(num_odd_dimensions.flatten())
            img_sample_sz = mround(new_img_sample_sz + best_choice)

        self.sample_sz = img_sample_sz
        self.data_sz = [img_sample_sz // self._cell_size]
        return img_sample_sz

    def _sample_patch(self, im, pos, sample_sz, output_sz):
        pos = np.floor(pos)
        sample_sz = np.maximum(mround(sample_sz), 1)
        xs = np.floor(pos[1]) + np.arange(0, sample_sz[1]+1) - np.floor((sample_sz[1]+1)/2)
        ys = np.floor(pos[0]) + np.arange(0, sample_sz[0]+1) - np.floor((sample_sz[0]+1)/2)
        xmin = max(0, int(xs.min()))
        xmax = min(im.shape[1], int(xs.max()))
        ymin = max(0, int(ys.min()))
        ymax = min(im.shape[0], int(ys.max()))
        # extract image
        im_patch = im[ymin:ymax, xmin:xmax, :]
        left = right = top = down = 0
        if xs.min() < 0:
            left = int(abs(xs.min()))
        if xs.max() > im.shape[1]:
            right = int(xs.max() - im.shape[1])
        if ys.min() < 0:
            top = int(abs(ys.min()))
        if ys.max() > im.shape[0]:
            down = int(ys.max() - im.shape[0])
        if left != 0 or right != 0 or top != 0 or down != 0:
            im_patch = cv2.copyMakeBorder(im_patch, top, down, left, right, cv2.BORDER_REPLICATE)
        
        # im_patch = cv2.resize(im_patch, (int(output_sz[0]), int(output_sz[1])))
        im_patch = cv2.resize(im_patch, (int(output_sz[0]), int(output_sz[1])), cv2.INTER_CUBIC)
        if len(im_patch.shape) == 2:
            im_patch = im_patch[:, :, np.newaxis]
        return im_patch

    def _feature_normalization(self, x):
        if hasattr(config, 'normalize_power') and config.normalize_power > 0:
            if config.normalize_power == 2:
                x = x * np.sqrt((x.shape[0]*x.shape[1]) ** config.normalize_size * (x.shape[2]**config.normalize_dim) / (x**2).sum(axis=(0, 1, 2)))
            else:
                x = x * ((x.shape[0]*x.shape[1]) ** config.normalize_size) * (x.shape[2]**config.normalize_dim) / ((np.abs(x) ** (1. / config.normalize_power)).sum(axis=(0, 1, 2)))

        if config.square_root_normalization:
            x = np.sign(x) * np.sqrt(np.abs(x))
        return x.astype(np.float32)

class DeepHeatmapFeature(Feature):
    def __init__(self, fname, compressed_dim):
        super().__init__()

        self._compressed_dim = compressed_dim

    def init_size(self, img_sample_sz, bbox, frame_shape, decoder_shape, cell_size=None):
        img_sample_sz = img_sample_sz.astype(np.int32)
        
        feat_shape = np.ceil(img_sample_sz / 16)
        
        desired_sz = feat_shape + 1 + feat_shape % 2
        img_sample_sz = desired_sz * 16

        self.num_dim = [16]
        self.sample_sz = img_sample_sz

        B, C, H, W = decoder_shape
        img_h, img_w, _ = frame_shape

        # case 1: h=20, img_h=2160, H=544 -> data_sz=12
        # case 2: h=20, img_h=1080, H=544 -> data_sz=24

        scale = H / img_h
        data_sz = int(img_sample_sz[0] * scale)

        self.data_sz = [np.array([data_sz, data_sz])]

        return img_sample_sz

    def get_features(self, img, pos, sample_sz, scales, decoder_output):
        features = decoder_output + 1e-6

        if not isinstance(scales, list) and not isinstance(scales, np.ndarray):
            scales = [scales]

        # (1, 16, 544, 960)
        B, C, H, W = features.shape
        img_h, img_w, _ = img.shape

        features = features[0].transpose(1, 2, 0)

        scale_h = H / img_h
        scale_w = W / img_w

        pos = pos * np.array([scale_h, scale_w])
        sample_sz = sample_sz * np.array([scale_h, scale_w])

        patches = []
        for scale in scales:
            patch = self._sample_patch(features, pos, sample_sz*scale, sample_sz)      
            patches.append(patch)

        patches = np.stack(patches, axis=0).transpose(1, 2, 3, 0)
        f1 = self._feature_normalization(patches)

        return [f1]



class CNNFeature(Feature):
    def _forward(self, x):
        pass

    def get_features(self, img, pos, sample_sz, scales, _):
        if img.shape[2] == 1:
            img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2RGB)
            
        if not isinstance(scales, list) and not isinstance(scales, np.ndarray):
            scales = [scales]
            
        patches = []
        for scale in scales:
            patch = self._sample_patch(img, pos, sample_sz*scale, sample_sz)
            patch = mx.nd.array(patch / 255., ctx=self._ctx)
            normalized = mx.image.color_normalize(patch,
                                                  mean=mx.nd.array([0.485, 0.456, 0.406], ctx=self._ctx),
                                                  std=mx.nd.array([0.229, 0.224, 0.225], ctx=self._ctx))
            normalized = normalized.transpose((2, 0, 1)).expand_dims(axis=0)
            patches.append(normalized)
    
        patches = mx.nd.concat(*patches, dim=0)
        f1, f2 = self._forward(patches)
        f1 = self._feature_normalization(f1)
        f2 = self._feature_normalization(f2)

        return f1, f2

class ResNet50Feature(CNNFeature):
    def __init__(self, fname, compressed_dim):
        self._ctx = mx.gpu(config.gpu_id) if config.use_gpu else mx.cpu(0)
        self._resnet50 = vision.resnet50_v2(pretrained=True, ctx = self._ctx)
        self._compressed_dim = compressed_dim
        self._cell_size = [4, 16]
        self.penalty = [0., 0.]
        self.min_cell_size = np.min(self._cell_size)

    def init_size(self, img_sample_sz, bbox=None, frame_shape=None, decoder_shape=None, cell_size=None):
        # only support img_sample_sz square
        img_sample_sz = img_sample_sz.astype(np.int32)
        
        feat1_shape = np.ceil(img_sample_sz / 4)
        feat2_shape = np.ceil(img_sample_sz / 16)
        
        desired_sz = feat2_shape + 1 + feat2_shape % 2
 
        img_sample_sz = desired_sz * 16
        self.num_dim = [64, 1024]
        self.sample_sz = img_sample_sz
        self.data_sz = [np.ceil(img_sample_sz / 4),
                        np.ceil(img_sample_sz / 16)]
        
        return img_sample_sz

    def _forward(self, x):
        # stage1
        bn0 = self._resnet50.features[0].forward(x)
        conv1 = self._resnet50.features[1].forward(bn0)     # x2
        bn1 = self._resnet50.features[2].forward(conv1)
        relu1 = self._resnet50.features[3].forward(bn1)
        pool1 = self._resnet50.features[4].forward(relu1)   # x4
        # stage2
        stage2 = self._resnet50.features[5].forward(pool1)  # x4
        stage3 = self._resnet50.features[6].forward(stage2) # x8
        stage4 = self._resnet50.features[7].forward(stage3) # x16
        
        return [pool1.asnumpy().transpose(2, 3, 1, 0),
                stage4.asnumpy().transpose(2, 3, 1, 0)]
