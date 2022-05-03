import numpy
import matplotlib.pyplot as plt
import json
import argparse


if __name__ == "__main__":

    # Add command line arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-l", "--load", help="Load json file containing stats", type=str, default=None)
    args = parser.parse_args()
    json_fname = 'dqn_100_stats.json'
    # json_fname = args.load
    
    with open(json_fname, 'r') as f:
        stats = json.load(f)
    
    t = range(0, len(stats['lengths']))
    fig, ax = plt.subplots(1, 3, figsize=(20,5))
    ax[0].plot(t, stats['lengths'], 'r')
    ax[0].set(xlabel='Epochs', ylabel='Game length')
    
    ax[1].plot(t, stats['scores'], 'g')
    ax[1].set(xlabel='Epochs', ylabel='Score')
    
    ax[2].plot(t, stats['times'], 'b')
    ax[2].set(xlabel='Epochs', ylabel='Epoch duration')