import numpy as np
import matplotlib.pyplot as plt
import json
import argparse


def summarizeList(L, n):
    L1 = []
    for i in range(len(L)//n):
        L1.append(np.mean(L[i*n:i*(n+1)]))
    return L1


if __name__ == "__main__":

    # Add command line arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-l", "--load", help="Load json file containing stats", type=str, default=None)
    args = parser.parse_args()
    json_fname = 'models/dqn_50000_stats.json'
    # json_fname = args.load
    
    with open(json_fname, 'r') as f:
        stats = json.load(f)
    
    interval = 100
    t = range(0, len(stats['lengths']), interval)
    fig, ax = plt.subplots(1, 3, figsize=(20,5))
    lengths = summarizeList(stats['lengths'], interval)
    scores = summarizeList(stats['scores'], interval)
    times = summarizeList(stats['times'], interval)
    
    
    ax[0].plot(t, lengths, 'r')
    ax[0].set(xlabel='Epochs', ylabel='Game length')
    
    ax[1].plot(t, scores, 'g')
    ax[1].set(xlabel='Epochs', ylabel='Score')
    
    ax[2].plot(t, times, 'b')
    ax[2].set(xlabel='Epochs', ylabel='Epoch duration')