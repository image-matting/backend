import cv2
import numpy as np
import torch
from torchvision import transforms


def inference_img_whole(args, model, img, trimap):
    h, w, c = img.shape
    new_h = min(args.max_size, h - (h % 32))
    new_w = min(args.max_size, w - (w % 32))

    scale_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scale_trimap = cv2.resize(trimap, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pred_mattes = inference_once(args, model, scale_img, scale_trimap, aligned=False)

    origin_pred_mattes = cv2.resize(pred_mattes, (w, h), interpolation = cv2.INTER_LINEAR)
    assert(origin_pred_mattes.shape == trimap.shape)

    return origin_pred_mattes


def inference_once(args, model, scale_img, scale_trimap, aligned=True):
    if aligned:
        assert(scale_img.shape[0] == args.size_h)
        assert(scale_img.shape[1] == args.size_w)

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
    ])

    scale_img_rgb = cv2.cvtColor(scale_img, cv2.COLOR_BGR2RGB)
    tensor_img = normalize(scale_img_rgb).unsqueeze(0)

    tensor_trimap = torch.from_numpy(scale_trimap.astype(np.float32)[np.newaxis, np.newaxis, :, :])

    if args.cuda:
        tensor_img = tensor_img.cuda()
        tensor_trimap = tensor_trimap.cuda()

    input_t = torch.cat((tensor_img, tensor_trimap / 255.), 1)

    if args.stage <= 1:
        pred_mattes, _ = model(input_t)
    else:
        _, pred_mattes = model(input_t)
    pred_mattes = pred_mattes.data
    if args.cuda:
        pred_mattes = pred_mattes.cpu()
    pred_mattes = pred_mattes.numpy()[0, 0, :, :]

    return pred_mattes
