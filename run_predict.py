import os
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import utils
from model import unet


def get_args():
    parser = argparse .ArgumentParser('Ship track scripts', add_help=False)
    parser.add_argument('--predict_data', default='predict_data/', type=str)
    parser.add_argument('--predict_save_dir', default='images/predict_imgs/', type=str)
    parser.add_argument('--img_crop_size', default=512, type=int)
    parser.add_argument('--model_path', default='models/Unet_033-0.0065.hdf5', type=str)
    return parser.parse_args()


def get_model(path):
    model = unet(channels=2)
    model.load_weights(path)
    return model


def load_data(path):
    zeros = np.zeros((2048, 1536, 2))
    var1, var2, lon, lat = utils.read_hdf(path)

    var1 = var1[..., np.newaxis]
    var2 = var2[..., np.newaxis]
    X = np.concatenate([var1, var2], axis=-1)
    shape = np.shape(X)

    # split it so our model can do pooling
    zeros[:shape[0], :shape[1], :shape[2]] = X
    Xs = utils.split_array(zeros)
    out = np.stack(Xs)
    return out, var1.squeeze(), var2.squeeze(), lon, lat


def main(args):
    # load model
    model = get_model(args.model_path)

    # predict data
    paths = list(Path(args.predict_data).glob('*.hdf'))
    for path in paths:
        file_name = str(path).split("/")[-1].rstrip(".hdf")
        save_path = f'{args.predict_save_dir}{file_name}'

        # load data (var1, var2, lon, lat for plot)
        X, var1, var2, lon, lat = load_data(str(path))

        # predict
        predict = model.predict(X, batch_size=2)
        predict = predict.squeeze() > 0.001
        out = utils.concat_arr2d(predict)
        out = out[:lon.shape[0], :lon.shape[1]]

        # visulization
        utils.ship_track_vis(var1, var2, out, lon, lat, save_path)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    opts = get_args()
    main(opts)
