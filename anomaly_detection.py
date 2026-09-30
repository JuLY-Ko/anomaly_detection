import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def detect(x:np.ndarray, window: int=500, q:float=90.0):
    if not 2<= window <= len(x):
        raise ValueError('window must be between 2 and the number of rows')
    if not 0<= q <= 100:
        raise ValueError('q mist be in [0,100]')
    
    predictions = np.zeros(len(x), dtype=bool)
    thresholds = np.full(len(x), np.nan)

    first_thres = np.percentile(x[:window],q, method='linear') # first threshold calculation & label the first window

    predictions[:window] = x[:window] >= first_thres
    thresholds[:window] = first_thres

    for end in range(window, len(x)): # move one row and label new point
        threshold = np.percentile(x[end-window+1:end+1],q,method='linear')
        predictions[end]= x[end] >= threshold
        thresholds[end] = threshold

    return predictions, thresholds

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv', nargs='?', default="AG_NO3_fill_cells_remove_NAN-2.csv")
    p.add_argument('--window', type=int, default = 500)
    p.add_argument('--q',type=float, default=90.0)
    p.add_argument('--output', type=Path, default=Path('results'))
    args= p.parse_args()

    df=pd.read_csv(args.csv) # load cleaned csv & check the columns

    required = {'NO3N','Student_Flag'}
    if not required.issubset(df.columns):
        raise ValueError('Missing columns')
    if df[list(required)].isna().any().any():
        raise ValueError('Missing required column. use cleaned dataset')

    x = df['NO3N'].to_numpy(dtype=float)#nirate for detection

    truth= df['Student_Flag'].to_numpy(dtype=int)#flag for evaluation

    if not np.isfinite(x).all or not np.isin(truth,[0,1]).all():
        raise ValueError('NO3N must be finite. Flag must conatin only 0 and 1')
    
    pred, thresholds= detect(x, args.window, args.q)

    actual = truth == 1 
    tp =int(np.sum(pred & actual))
    fp= int(np.sum(pred & ~actual))
    fn= int(np.sum(~pred & actual))
    tn= int(np.sum(~pred & ~actual))

    args.output.mkdir(parents=True, exist_ok=True) #save each prediction & thresholds
    result = df.copy()
    result['threshold']=thresholds
    result['anomaly_predicted'] = pred.astype(int)
    result.to_csv(args.output /'predictions.csv', index=False)

    print(f'W={args.window}, q={args.q:g}; TP={tp}, FP={fp}, FN={fn}, TN={tn}')
    print(f'Normal accuracy: {tn/(tn+fp):.4%}')
    print(f'Anormaly accuracy: {tp/(tp+fn):.4%}')

    fig, ax = plt. subplots(figsize=(14,5), constrained_layout=True)

    index = np.arange(len(x))

    ax.plot(index, x, lw=0.6, color='red', label='NO3N')

    marked = index[pred]
    ax.scatter(marked, x[marked],s=11, color = 'blue', label='Predicted anomaly', zorder = 3)

    ax.set(xlabel='Obervation index', ylabel = 'NO3N', title = 'Full nitrate series')
    ax.legend(loc='upper right')
    
    fig.savefig(args.output / 'detected_anomalies.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()



