import numpy as np
import pandas as pd
import h5py

import tensorflow as tf
from tfbio.data import Featurizer, make_grid

import os


def input_file(path):
    """Check if input file exists."""

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError('File %s does not exist.' % path)
    return path


def network_prefix(path):
    """Check if all file required to restore the network exists."""

    from glob import glob
    dir_path, file_name = os.path.split(path)
    path = os.path.join(os.path.abspath(dir_path), file_name)

    for extension in ['index', 'meta', 'data*']:
        file_name = '%s.%s' % (path, extension)

        # use glob instead of os because we need to expand the wildcard
        if len(glob(file_name)) == 0:
            raise IOError('File %s does not exist.' % file_name)

    return path


def batch_size(value):
    """Check if batch size is a non-negative integer"""

    value = int(value)
    if value < 0:
        raise ValueError('Batch size must be positive, %s given' % value)
    return value


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
    description='Predict affinity with the network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    epilog='''This script reads the structures of complexes from HDF file and
    predicts binding affinity for each comples. The input can be prepared with
    prepare.py script. If you want to prepare the data and run the model manualy
    use functions defined in utils module.
    '''
)

parser.add_argument('--input', '-i', required=True, type=input_file,
                    help='HDF file with prepared structures')
parser.add_argument('--network', '-n', type=network_prefix,
                    default='results/batch5-2017-06-05T07:58:47-best',
                    help='prefix for the files with the network'
                    'Be default we use network trained on PDBbind v. 2016')
parser.add_argument('--grid_spacing', '-g', default=1.0, type=float,
                    help='distance between grid points used during training')
parser.add_argument('--max_dist', '-d', default=10.0, type=float,
                    help='max distance from complex center used during training')
parser.add_argument('--batch', '-b', type=batch_size,
                    default=20,
                    help='batch size. If set to 0, predict for all complexes at once.')
parser.add_argument('--charge_scaler', type=float, default=0.425896,
                    help='scaling factor for the charge'
                         ' (use the same factor when preparing data for'
                         ' training and and for predictions)')
parser.add_argument('--output', '-o', type=output_file,
                    default='./predictions.csv',
                    help='name for the CSV file with the predictions')
parser.add_argument('--verbose', '-v', type=string_bool,
                    default=True,
                    help='whether to print messages')


args = parser.parse_args()

# TODO: avarage prediction for different rotations (optional)

featurizer = Featurizer()

charge_column = featurizer.FEATURE_NAMES.index('partialcharge')

coords = []
features = []
names = []

with h5py.File(args.input, 'r') as f:
    for name in f:
        names.append(name)
        dataset = f[name]
        coords.append(dataset[:, :3])
        features.append(dataset[:, 3:])


if args.verbose:
    print('loaded %s complexes\n' % len(coords))


def __get_batch():

    batch_grid = []

    if args.verbose:
        if args.batch == 0:
            print('predict for all complexes at once\n')
        else:
            print('%s samples per batch\n' % args.batch)

    for crd, f in zip(coords, features):
        batch_grid.append(make_grid(crd, f, max_dist=args.max_dist,
                          grid_resolution=args.grid_spacing))
        if len(batch_grid) == args.batch:
            # if batch is not specified it will never happen
            batch_grid = np.vstack(batch_grid)
            batch_grid[..., charge_column] /= args.charge_scaler
            yield batch_grid
            batch_grid = []

    if len(batch_grid) > 0:
        batch_grid = np.vstack(batch_grid)
        batch_grid[..., charge_column] /= args.charge_scaler
        yield batch_grid


saver = tf.train.import_meta_graph('%s.meta' % args.network,
                                   clear_devices=True)


predict = tf.get_collection('output')[0]
inp = tf.get_collection('input')[0]
kp = tf.get_collection('kp')[0]

if args.verbose:
    print('restored network from %s\n' % args.network)

with tf.Session() as session:
    saver.restore(session, args.network)
    predictions = []
    batch_generator = __get_batch()
    for grid in batch_generator:
        # TODO: remove kp in next release
        # it's here for backward compatibility
        predictions.append(session.run(predict, feed_dict={inp: grid, kp: 1.0}))

results = pd.DataFrame({'name': names,
                        'prediction': np.vstack(predictions).flatten()})
results.to_csv(args.output, index=False)
if args.verbose:
    print('results saved to %s' % args.output)
