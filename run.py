import os
import sys
import argparse
import numpy as np

from generator_model import GeneratorModel as ICVAE
from theano_ops.optimizers import Adam


def main(argv):
    parser = argparse.ArgumentParser(description='Trains the ICVAE model')
    parser.add_argument('--data_root', type=str, help='Path to where both real and fake npy data are stored', required=True)
    parser.add_argument('--init_params', type=str, help='Path to the initial parameters', required=False, default=None)
    parser.add_argument('--nb_epochs', type=int, help='Number of training epochs', required=False, default=100000)
    parser.add_argument('--batch_size', type=int, help='Batch size', required=False, default=100)
    parser.add_argument('--lr', type=float, help='Learning rate', required=False, default=1e-5)
    parser.add_argument('--mode', type=str, help='Run in train or generate modes', required=False, default='train')
    
    args = parser.parse_args()

    data_root = args.data_root
    x_train = np.load(os.path.join(data_root, 'X_Train_Real_HQ.npy'))
    y_train = np.load(os.path.join(data_root, 'Y_Train_Real_HQ.npy'))

    x_train = np.vstack((np.load(os.path.join(data_root, 'X_Train_Fake_HQ.npy')), x_train))
    y_train = np.vstack((np.load(os.path.join(data_root, 'Y_Train_Fake_HQ.npy')), y_train))

    x_train = np.expand_dims(x_train, 1)
    y_train = np.expand_dims(y_train, 1)
    
    params = None
    if args.init_params:
        assert os.path.exists(args.init_params)
        params = np.load(args.init_params)
        
    G = ICVAE(batch_size=args.batch_size,
              input_shape=(1, 1,128, 128),
              optimizer=Adam(lr=args.lr),
              metrics=['loss'],
              lmbd=1e-6,
              init_params=params)
    if args.mode == 'train':
        G.train(x_train.astype('float32'), y_train.astype('uint8'), nb_epochs=args.nb_epochs)

if __name__ == "__main__":
    main(sys.argv[1:])
