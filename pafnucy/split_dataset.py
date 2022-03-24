from sklearn.utils import shuffle
import h5py

import os

import argparse

parser = argparse.ArgumentParser(description='Split dataset int training,'
                                 ' validation and test sets.')
parser.add_argument('--input_path', '-i', default='../pdbbind/v2016/',
                    help='directory with pdbbind dataset')
parser.add_argument('--output_path', '-o', default='../pdbbind/v2016/',
                    help='directory to store output files')
parser.add_argument('--size_val', '-s', type=int, default=1000,
                    help='number of samples in the validation set')
args = parser.parse_args()

# create files with the training and validation sets
with h5py.File('%s/training_set.hdf' % args.output_path, 'w') as g, \
     h5py.File('%s/validation_set.hdf' % args.output_path, 'w') as h:
    with h5py.File('%s/refined.hdf' % args.input_path, 'r') as f:
        refined_shuffled = shuffle(list(f.keys()), random_state=123)
        for pdb_id in refined_shuffled[:args.size_val]:
            ds = h.create_dataset(pdb_id, data=f[pdb_id])
            ds.attrs['affinity'] = f[pdb_id].attrs['affinity']
        for pdb_id in refined_shuffled[args.size_val:]:
            ds = g.create_dataset(pdb_id, data=f[pdb_id])
            ds.attrs['affinity'] = f[pdb_id].attrs['affinity']
    with h5py.File('%s/general.hdf' % args.input_path, 'r') as f:
        for pdb_id in f:
            ds = g.create_dataset(pdb_id, data=f[pdb_id])
            ds.attrs['affinity'] = f[pdb_id].attrs['affinity']

# create a symlink for the test set
os.symlink(os.path.abspath('%s/core.hdf' % args.input_path),
           os.path.abspath('%s/test_set.hdf' % args.output_path))
