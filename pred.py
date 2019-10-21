from __future__ import print_function, unicode_literals
import argparse
from tqdm import tqdm

from utils.fh_utils import *


def main(base_path, pred_out_path, pred_func, set_name=None):
    """
        Main eval loop: Iterates over all evaluation samples and saves the corresponding predictions.
    """
    # default value
    if set_name is None:
        set_name = 'evaluation'

    # init output containers
    xyz_pred_list, verts_pred_list = list(), list()

    K_list = json_load(os.path.join(base_path, '%s_K.json' % set_name))
    scale_list = json_load(os.path.join(base_path, '%s_scale.json' % set_name))

    # iterate over the dataset once
    for idx in tqdm(range(db_size(set_name))):
        if idx >= db_size(set_name):
            break

        # load input image
        img = read_img(idx, base_path, set_name)

        # use some algorithm for prediction
        xyz, verts = pred_func(
            img,
            np.array(K_list[idx]),
            scale_list[idx]
        )
        xyz_pred_list.append(xyz)
        verts_pred_list.append(verts)

    # dump results
    dump(pred_out_path, xyz_pred_list, verts_pred_list)


def dump(pred_out_path, xyz_pred_list, verts_pred_list):
    """ Save predictions into a json file. """
    # make sure its only lists
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]

    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump(
            [
                xyz_pred_list,
                verts_pred_list
            ], fo)
    print('Dumped %d joints and %d verts predictions to %s' % (len(xyz_pred_list), len(verts_pred_list), pred_out_path))


def pred_template(img, K, scale):
    """ Predict joints and vertices from a given sample.
        img: (224, 224, 30 RGB image.
        K: (3, 3) camera intrinsic matrix.
        scale: () scalar metric length of the reference bone,
                  which was calculated as np.linalg.norm(xyz[9] - xyz[10], 2),
                  i.e. it is the length of the proximal phalangal bone of the middle finger.
    """
    # TODO: Put your algorithm here, which computes (metric) 3D joint coordinates and 3D vertex positions
    xyz = np.zeros((21, 3))  # 3D coordinates of the 21 joints
    verts = np.zeros((778, 3)) # 3D coordinates of the shape vertices
    return xyz, verts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('base_path', type=str,
                        help='Path to where the FreiHAND dataset is located.')
    parser.add_argument('--out', type=str, default='pred.json',
                        help='File to save the predictions.')
    args = parser.parse_args()

    # call with a predictor function
    main(
        args.base_path,
        args.out,
        pred_func=pred_template,
        set_name='evaluation'
    )

