import torch
import argparse
import net
import cv2
import os
import numpy as np
from predictor import inference_img_whole
import pickle

'''
    Load model exported by python2 with python3 will cause error:
        ascii' codec can't decode byte 0xda in position 5
    The following codes will fix the bug
'''


class AlphaMaskGenerator:
    _STAGE = 1

    def __init__(self, model_dir):
        parser = argparse.ArgumentParser()
        self.args = parser.parse_args()
        self.args.cuda = False
        self.args.stage = 1
        self.args.crop_or_resize = "whole"
        self.args.max_size = 1600

        self.model = net.VGG16(self._STAGE)
        ckpt = _my_torch_load(model_dir)
        self.model.load_state_dict(ckpt['state_dict'], strict=True)

    def generate_alpha_mask(self, input_image_path, trimap_image_path):
        input_image = cv2.imread(input_image_path)
        trimap_image = cv2.imread(trimap_image_path)[:, :, 0]

        torch.cuda.empty_cache()
        with torch.no_grad():
            pred_mattes = inference_img_whole(self.args, self.model, input_image, trimap_image)

        pred_mattes = (pred_mattes * 255).astype(np.uint8)
        pred_mattes[trimap_image == 255] = 255
        pred_mattes[trimap_image == 0] = 0

        return pred_mattes


def _my_torch_load(model_dir):
    try:
        ckpt = torch.load(model_dir)
        return ckpt
    except Exception as e:
        print("Load Error:{}\nTry Load Again...".format(e))

        class C:
            pass

        def c_load(ss):
            return pickle.load(ss, encoding='latin1')

        def c_unpickler(ss):
            return pickle.Unpickler(ss, encoding='latin1')

        c = C
        c.load = c_load
        c.Unpickler = c_unpickler
        ckpt = torch.load(model_dir, encoding='latin1')

        return ckpt
