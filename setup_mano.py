from __future__ import print_function, unicode_literals
import argparse
import os
from tempfile import mkstemp
from shutil import move
from os import fdopen, remove
import hashlib
import shutil

def replace(file_path, line_ids, new_lines):
    """ Replace a line in a given file with a new given line. """
    line_ids = [i - 1 for i in line_ids]
    # Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for i, line in enumerate(old_file):
                if i in line_ids:
                    new_file.write(new_lines[line_ids.index(i)] + '\n')
                else:
                    new_file.write(line)

    #Remove original file
    remove(file_path)

    #Move new file
    move(abs_path, file_path)

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def _patch_mano_loader():
    file = 'utils/mano_core/mano_loader.py'

    replace(file, *zip(*
        [
            (26, '    from posemapper import posemap'),
            (62, 'def load_model(fname_or_dict, ncomps=6, flat_hand_mean=False, v_template=None, use_pca=True):'),
            (66, '    from verts import verts_core'),
            (80, '    if use_pca:\n        hands_components = smpl_data[''hands_components'']  # PCA components\n    else:\n        hands_components = np.eye(45)  # directly modify 15x3 articulation angles'),
            (131,'    result.dd = dd'),
        ]
    ))


def patch_files():
    _patch_mano_loader()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Import needed files from MANO repository.')
    parser.add_argument('mano_path', type=str, help='Path to where the original MANO repository is located.')
    parser.add_argument('--clear', action='store_true', help='Util call for me to remove mano files before committing.')
    args = parser.parse_args()

    # files we attempt to copy from the original mano repository
    files_needed = [
        'models/MANO_RIGHT.pkl',
        'webuser/verts.py',
        'webuser/posemapper.py',
        'webuser/lbs.py',
        'webuser/smpl_handpca_wrapper_HAND_only.py',
    ]

    # how to rename them to be used in our repository
    files_copy_to = [
        'data/MANO_RIGHT.pkl',
        'utils/mano_core/verts.py',
        'utils/mano_core/posemapper.py',
        'utils/mano_core/lbs.py',
        'utils/mano_core/mano_loader.py',
    ]

    if args.clear:
        for f in files_copy_to:
            if os.path.exists(f):
                os.remove(f)
            if '.py' in f:
                f2 = f.replace('.py', '.pyc')
                if os.path.exists(f2):
                    os.remove(f2)
        print('Repository cleaned.')
        exit()

    # check input files
    files_needed = [os.path.join(args.mano_path, f) for f in files_needed]
    assert all([os.path.exists(f) for f in files_needed]), 'Could not find one of the needed MANO files in the directory you provided.'

    # coursely check content
    hash_ground_truth = [
        'fd5a9d35f914987cf1cc04ffe338caa1',
        '998c30fd83c473da6178aa2cb23b6a5d',
        'c5e9eacc535ec7d03060e0c8d6f80f45',
        'd11c767d5db8d4a55b4ece1c46a4e4ac',
        '5afc7a3eb1a6ce0c2dac1483338a5f58'
    ]
    assert all([md5(f) == gt for f, gt in zip(files_needed, hash_ground_truth)]), 'Hash sum of provided files differs from what was expected.'

    # copy files
    for a, b in zip(files_needed, files_copy_to):
        shutil.copy2(a, b)

    # some files need to be modified
    patch_files()

