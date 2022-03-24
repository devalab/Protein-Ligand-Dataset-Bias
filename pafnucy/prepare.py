import numpy as np
import pandas as pd
import h5py

import pybel
from tfbio.data import Featurizer

import os


def input_file(path):
    """Check if input file exists."""

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError('File %s does not exist.' % path)
    return path


def output_file(path):
    """Check if output file can be created."""

    path = os.path.abspath(path)
    dirname = os.path.dirname(path)

    if not os.access(dirname, os.W_OK):
        raise IOError('File %s cannot be created (check your permissions).'
                      % path)
    return path


def string_bool(s):
    s = s.lower()
    if s in ['true', 't', '1', 'yes', 'y']:
        return True
    elif s in ['false', 'f', '0', 'no', 'n']:
        return False
    else:
        raise IOError('%s cannot be interpreted as a boolean' % s)


import argparse
parser = argparse.ArgumentParser(
    description='Prepare molecular data for the network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    epilog='''This script reads the structures of ligands and pocket(s),
    prepares them for the neural network and saves in a HDF file.
    It also saves affinity values as attributes, if they are provided.
    You can either specify a separate pocket for each ligand or a single
    pocket that will be used for all ligands. We assume that your structures
    are fully prepared.\n\n

    Note that this scripts produces standard data representation for our network
    and saves all required data to predict affinity for each molecular complex.
    If some part of your data can be shared between multiple complexes
    (e.g. you use a single structure for the pocket), you can store the data
    more efficiently. To prepare the data manually use functions defined in
    tfbio.data module.
    '''
)

parser.add_argument('--ligand', '-l', required=True, type=input_file, nargs='+',
                    help='files with ligands\' structures')
parser.add_argument('--pocket', '-p', required=True, type=input_file, nargs='+',
                    help='files with pockets\' structures')
parser.add_argument('--ligand_format', type=str, default='mol2',
                    help='file format for the ligand,'
                         ' must be supported by openbabel')

parser.add_argument('--pocket_format', type=str, default='mol2',
                    help='file format for the pocket,'
                         ' must be supported by openbabel')
parser.add_argument('--output', '-o', default='./complexes.hdf',
                    type=output_file,
                    help='name for the file with the prepared structures')
parser.add_argument('--mode', '-m', default='w',
                    type=str, choices=['r+', 'w', 'w-', 'x', 'a'],
                    help='mode for the output file (see h5py documentation)')
parser.add_argument('--affinities', '-a', default=None, type=input_file,
                    help='CSV table with affinity values.'
                         ' It must contain two columns: `name` which must be'
                         ' equal to ligand\'s file name without extenstion,'
                         ' and `affinity` which must contain floats')
parser.add_argument('--verbose', '-v', default=True, type=string_bool,
                    help='whether to print messages')

args = parser.parse_args()


# TODO: training set preparation (allow to read affinities)


num_pockets = len(args.pocket)
num_ligands = len(args.ligand)
if num_pockets != 1 and num_pockets != num_ligands:
    raise IOError('%s pockets specified for %s ligands. You must either provide'
                  'a single pocket or a separate pocket for each ligand'
                  % (num_pockets, num_ligands))
if args.verbose:
    print('%s ligands and %s pockets to prepare:' % (num_ligands, num_pockets))
    if num_pockets == 1:
        print(' pocket: %s' % args.pocket[0])
        for ligand_file in args.ligand:
            print(' ligand: %s' % ligand_file)
    else:
        for ligand_file, pocket_file in zip(args.ligand, args.pocket):
            print(' ligand: %s, pocket: %s' % (ligand_file, pocket_file))
    print('\n\n')


if args.affinities is not None:
    affinities = pd.read_csv(args.affinities)
    if 'affinity' not in affinities.columns:
        raise ValueError('There is no `affinity` column in the table')
    elif 'name' not in affinities.columns:
        raise ValueError('There is no `name` column in the table')
    affinities = affinities.set_index('name')['affinity']
else:
    affinities = None

featurizer = Featurizer()


def __get_pocket():
    if num_pockets > 1:
        for pocket_file in args.pocket:
            if args.verbose:
                print('reading %s' % pocket_file)
            try:
                pocket = next(pybel.readfile(args.pocket_format, pocket_file))
            except:
                raise IOError('Cannot read %s file' % pocket_file)

            pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
            yield (pocket_coords, pocket_features)

    else:
        pocket_file = args.pocket[0]
        try:
            pocket = next(pybel.readfile(args.pocket_format, pocket_file))
        except:
            raise IOError('Cannot read %s file' % pocket_file)
        pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
        for _ in range(num_ligands):
            yield (pocket_coords, pocket_features)


with h5py.File(args.output, args.mode) as f:
    pocket_generator = __get_pocket()
    for ligand_file in args.ligand:
        # use filename without extension as dataset name
        name = os.path.splitext(os.path.split(ligand_file)[1])[0]

        if args.verbose:
            print('reading %s' % ligand_file)
        try:
            ligand = next(pybel.readfile(args.ligand_format, ligand_file))
        except:
            raise IOError('Cannot read %s file' % ligand_file)

        ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)
        pocket_coords, pocket_features = next(pocket_generator)

        centroid = ligand_coords.mean(axis=0)
        ligand_coords -= centroid
        pocket_coords -= centroid

        data = np.concatenate(
            (np.concatenate((ligand_coords, pocket_coords)),
             np.concatenate((ligand_features, pocket_features))),
            axis=1,
        )

        dataset = f.create_dataset(name, data=data, shape=data.shape,
                                   dtype='float32', compression='lzf')
        if affinities is not None:
            dataset.attrs['affinity'] = affinities.loc[name]
if args.verbose:
    print('\n\ncreated %s with %s structures' % (args.output, num_ligands))
