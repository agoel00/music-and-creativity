import creativity 
import numpy as np
import pandas as pd 
import pickle
import tqdm
import time

forms = {
    'angry': 'BRED_angry (Responses) - Form Responses 1.csv',
    'sad': 'BRED_sad (Responses) - Form Responses 1.csv',
    'relaxed': 'BRED_relaxed (Responses) - Form Responses 1.csv',
    'happy': 'BRED_happy (Responses) - Form Responses 1.csv',
    'control': 'BRED_control (Responses) - Form Responses 1.csv'
}

model = creativity.Model('glove.840B.300d.txt', 'words.txt')

# for key, val in forms.items():
#     forms[key] = val.replace('/edit#gid=', '/export?format=csv&gid=')

results = {}

def main():
    for key, val in tqdm.tqdm(forms.items()):
        df = pd.read_csv("data/{}".format(val))
        # time.sleep(1)
        dats = df.iloc[:, 5:15].values
        rats = df.iloc[:, 15:].values
        if key not in results:
            results[key] = {}    
        results[key]['dat'] = np.apply_along_axis(model.dat, 1, dats)
        results[key]['rat'] = np.apply_along_axis(model.rat, 1, rats)



    # for dat_ans, rat_ans in zip(dats, rats):
    #     print(dat_ans)
    #     print(rat_ans)
    #     results[key]['dat'].append(model.dat(dat_ans))
    #     results[key]['rat'].append(model.rat(rat_ans))

    print(results)

    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__=='__main__':
    main()