import argparse
import numpy as np
from data_loader import load_data
from train import train



if __name__ == '__main__':
    np.random.seed(666)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=4, help='dimension of embeddings')
    parser.add_argument('--n_hop', type=int, default=2, help='number of hops')
    parser.add_argument('--graph_weight', type=float, default=1e-2, help='weight of the graph term')
    parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
    parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
    parser.add_argument('--item_update_mode', type=str, default='plus_transform',help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True,help='whether using outputs of all hops or just the last hop when making prediction')
    args = parser.parse_args()
    train(args, load_data(args), False)
