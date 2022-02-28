from json.tool import main
import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import deque
import random
SEQ_LEN= 60
FUTURE_PERIOD_PREDICT= 3
RATIO_TO_PREDICT= "BTC-USD"

def preprocess_df(df):
    df = df.drop("future",axis=1)
    
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
            
    df.dropna(inplace = True) #just in case

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days),i[-1]])

    random.shuffle(sequential_data)

    buys = []
    sells = []
    
    for seq , target in sequential_data:
        if target == 0:
            sells.append([seq,target])
        elif target == 1:
            buys.append([seq,target])
    
    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys),len(sells))

    buys = buys[:lower]
    sells = sells[:lower]
    
    sequential_data = buys + sells
    
    random.shuffle(sequential_data) 

    X = []
    y = []
    
    for seq , target in sequential_data:
        X.append(seq)
        y.append(target)
        
    return np.array(X), y   

    
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


df = pd.read_csv("crypto_data/LTC-USD.csv",names=['time','low','high','open','close','volume'])

main_df = pd.DataFrame()

ratios = ["BTC-USD","LTC-USD","ETH-USD","BCH-USD"]

for ratio in ratios:
    dataset = f"crypto_data/{ratio}.csv"
    
    df = pd.read_csv(dataset, names=['time','low','high','open','close','volume'])

    df.rename(columns={"close": f"{ratio}_close", 
              "volume":f"{ratio}_volume"},
              inplace = True)

    df.set_index("time",inplace=True)

    df = df[[f"{ratio}_close",f"{ratio}_volume"]]

    if len(main_df) == 0:
        main_df = df
    
    else:
        main_df = main_df.join(df)

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = main_df.apply(lambda x: classify(x[f'{RATIO_TO_PREDICT}_close'],x["future"]),axis=1)

# print(main_df[[f"{RATIO_TO_PREDICT}_close","future","target"]].head(10))
times = sorted(main_df.index.values)

last_5pct = times[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x , train_y = preprocess_df(main_df)
validation_x , validation_y = preprocess_df(validation_main_df)

print((validation_y.count(0)),(validation_y.count(1)))
