import numpy as np

import cv2

from numpy.fft import fft, ifft
from scipy import signal
from .config import config
from .fourier_tools import resize_dft
from .features import fhog

import ipdb as pdb

class ScaleFilter:
    def __init__(self, target_sz, ):
        init_target_sz = target_sz
        num_scales = config.number_of_scales_filter
        scale_step = config.scale_step_filter
        scale_sigma = config.number_of_interp_scales * config.scale_sigma_factor

        scale_exp = np.arange(-np.floor(num_scales - 1)/2, np.ceil(num_scales-1)/2+1) * config.number_of_interp_scales / num_scales
        scale_exp_shift = np.roll(scale_exp, (0, -int(np.floor((num_scales-1)/2))))

        interp_scale_exp = np.arange(-np.floor((config.number_of_interp_scales-1)/2), np.ceil((config.number_of_interp_scales-1)/2)+1)
        interp_scale_exp_shift = np.roll(interp_scale_exp, [0, -int(np.floor(config.number_of_interp_scales-1)/2)])

        self.scale_size_factors = scale_step ** scale_exp
        self.interp_scale_factors = scale_step ** interp_scale_exp_shift

        ys = np.exp(-0.5 * (scale_exp_shift ** 2) / (scale_sigma ** 2))
        self.yf = fft(ys)# .astype(np.float32)
        self.window = signal.hann(ys.shape[0])

        # make sure the scale model is not to large, to save computation time
        if config.scale_model_factor**2 * np.prod(init_target_sz) > config.scale_model_max_area:
            config.scale_model_factor = np.sqrt(config.scale_model_max_area / np.prod(init_target_sz))

        # set the scale model size
        config.scale_model_sz = np.maximum(np.floor(init_target_sz * config.scale_model_factor), np.array([8, 8]))
        self.max_scale_dim = config.s_num_compressed_dim == 'MAX'
        if self.max_scale_dim:
            config.s_num_compressed_dim = len(self.scale_size_factors)

        self.num_scales = num_scales
        self.scale_step = scale_step
        self.scale_factors = np.array([1])

    def track(self, im, pos, base_target_sz, current_scale_factor):
        """
            track the scale using the scale filter
        """
        # get scale filter features
        scales =  current_scale_factor * self.scale_size_factors
        xs = self._extract_scale_sample(im, pos, base_target_sz, scales, config.scale_model_sz)

        # project
        xs = self._feature_proj_scale(xs, self.basis, self.window)

        # get scores
        xsf = fft(xs, None, 1)
        scale_responsef = np.sum(self.sf_num * xsf, 0) / (self.sf_den + config.lamBda)
        interp_scale_response = np.real(ifft(resize_dft(scale_responsef, config.number_of_interp_scales)))

        recovered_scale_index = np.argmax(interp_scale_response)# == np.max(interp_scale_response)

        if config.do_poly_interp:
            # fit a quadratic polynomial to get a refined scale estimate
            id1 = (recovered_scale_index - 1 - 1) % config.number_of_interp_scales
            id2 = (recovered_scale_index + 1 - 1) % config.number_of_interp_scales

            poly_x = np.array([self.interp_scale_factors[id1], self.interp_scale_factors[recovered_scale_index], self.interp_scale_factors[id2]])
            poly_y = np.array([interp_scale_response[id1], interp_scale_response[recovered_scale_index], interp_scale_response[id2]])
            poly_A = np.array([[poly_x[0]**2, poly_x[0], 1],
                               [poly_x[1]**2, poly_x[1], 1],
                               [poly_x[2]**2, poly_x[2], 1]])
            poly = np.linalg.inv(poly_A).dot(poly_y.T)
            scale_change_factor = - poly[1] / (2 * poly[0])
        else:
            scale_change_factor = self.interp_scale_factors[recovered_scale_index]
        return scale_change_factor

    def update(self, im, pos, base_target_sz, current_scale_factor):
        """
            update the scale filter
        """
        # get scale filter features
        scales = current_scale_factor * self.scale_size_factors
        xs = self._extract_scale_sample(im, pos, base_target_sz, scales, config.scale_model_sz)

        first_frame = not hasattr(self, 's_num')

        if first_frame:
            self.s_num = xs
        else:
            self.s_num = (1 - config.scale_learning_rate) * self.s_num + config.scale_learning_rate * xs
        # compute projection basis
        bigY = self.s_num
        bigY_den = xs
        if self.max_scale_dim:
            self.basis, _ = np.linalg.qr(bigY)
            scale_basis_den, _ = np.linalg.qr(bigY_den)
        else:
            U, _, _ = np.linalg.svd(bigY)
            self.basis = U[:, :config.s_num_compressed_dim]
        self.basis = self.basis.T

        # compute numerator
        sf_proj = fft(self._feature_proj_scale(self.s_num, self.basis, self.window), None, 1)
        self.sf_num = self.yf * np.conj(sf_proj)

        # update denominator
        xs = self._feature_proj_scale(xs, scale_basis_den.T, self.window)
        xsf = fft(xs, None, 1)
        new_sf_den = np.sum(xsf * np.conj(xsf), 0)
        if first_frame:
            self.sf_den = new_sf_den
        else:
            self.sf_den = (1 - config.scale_learning_rate) * self.sf_den + config.scale_learning_rate * new_sf_den

    def _extract_scale_sample(self, im, pos, base_target_sz, scale_factor, scale_model_sz):
        num_scales = len(scale_factor)

        # downsample factor
        df = np.floor(np.min(scale_factor))
        if df > 1:
            im = im[::df, ::df]
            pos = (pos - 1) / df + 1
            scale_factor = scale_factor / df

        scale_sample = []
        for idx, scale in enumerate(scale_factor):
            patch_sz = np.floor(base_target_sz * scale)

            xmin = int(max(0, np.floor(pos[1] - np.floor((patch_sz[1]+1)/2))))
            xmax = int(min(im.shape[1], np.floor(pos[1] + np.floor((patch_sz[1]+1)/2))))
            ymin = int(max(0, np.floor(pos[0] - np.floor((patch_sz[0]+1)/2))))
            ymax = int(min(im.shape[0], np.floor(pos[0] + np.floor((patch_sz[0]+1)/2))))
            # check for out-of-bounds coordinates, and set them to the values at the borders

            # extract image
            im_patch = im[ymin:ymax, xmin:xmax :]
            # if im.shape[1] > scale_model_sz[0]:
            #     interpolation = cv2.INTER_LINEAR
            # else:
            #     interpolation = cv2.INTER_AREA

            # resize image to model size
            im_patch_resized = cv2.resize(im_patch,
                                          (int(scale_model_sz[0]),int(scale_model_sz[1])))
                                          # interpolation)

            # extract scale features
            scale_sample.append(fhog(im_patch_resized, 4)[:, :, :31].reshape(-1, 1))

        scale_sample = np.concatenate(scale_sample, axis=1)
        return scale_sample

    def _feature_proj_scale(self, x, proj_matrix, cos_window):
        return proj_matrix.dot(x) * cos_window
