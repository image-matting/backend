import numpy as np
import torch
from skimage import transform, io
from torch.autograd import Variable

from model.u2net import U2NET_full


class U2NetSalientObjectDetector:
    def __init__(self, model_dir):
        self.u2net = _load_u2net(model_dir)

    def detect(self, input_image_path):
        input_test = _load_single_input_test(input_image_path)

        d1, d2, d3, d4, d5, d6, d7 = self.u2net(input_test)

        pred = d1[:, 0, :, :]
        pred = _normalize(pred)

        return pred


def _load_u2net(model_dir):
    u2net = U2NET_full()
    if torch.cuda.is_available():
        u2net.load_state_dict(torch.load(model_dir))
        u2net.cuda()
    else:
        u2net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    u2net.eval()

    return u2net


def _load_single_input_test(image_path):
    image = io.imread(image_path)
    image = transform.resize(image, (320, 320), mode='constant')

    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
    tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
    tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
    tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

    tmpImg = tmpImg.transpose((2, 0, 1))
    image = torch.from_numpy(np.array([tmpImg]))

    input_test = image.type(torch.FloatTensor)
    input_test = Variable(input_test)

    return input_test


def _normalize(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn
