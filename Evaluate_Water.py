"""
NIPA2022 - task10_water
Nyeongmin Lee

RMSE - by location & factor,
       only for factor/date existing in answer
       
<answer>
answer.npy // answer values, 3D array
mask.npy // NaN mask, binary 3D array
public.npy // public data mask, binary 3D array
private.npy // private data mask, binary 3D array

<sample>
{'location1': {
    {'yyyymmdd': {'factor1': xxx, 'factor2':xxx,...},
     'yyyymmdd': ...
    },
    {'yyyymmdd': {'factor1': xxx, 'factor2':xxx,...},
     'yyyymmdd': ...
    },
    ...
}
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error



def load_json(path):
    try:
        with open(path, 'r', encoding='utf8') as f:
            return json.load(f)

    except:
        assert False, print("unloadable json file")

def load_result(dic):
    npys = []
    ref_date = ['{0}{1:02d}{2:02d}'.format(j.year,j.month,j.day) for j in list(pd.date_range(start='2018-02-01', end='2019-12-31', inclusive="both"))]
    
    for i,location in enumerate(dic):
        assert len(dic[location]) == len(ref_date), f'Missing some dates in {i}th location'
        
        npy = []
        factor_names = ['pH','COD','SS','N','P','T']
        
        for date, factors in dic[location].items():
            assert len(factors) == 6, f'not 6 factors in {location}'
            npy_temp = np.array([factors[i] for i in factor_names])
            assert not np.isnan(np.sum(npy_temp)), f'Missing some factors in {location}'
            npy.append(npy_temp)
            
        npys.append(np.transpose(npy))
        
    return np.asarray(npys)


def mse(answer_path, pred_path):
    
    a_npy = np.load(os.path.join(answer_path,'answer.npy')) # location * factors * date
    mask_npy = np.load(os.path.join(answer_path,'mask.npy')) # location * factors * date
    pub_npy = np.load(os.path.join(answer_path,'public.npy')) # location * date
    prv_npy = np.load(os.path.join(answer_path,'private.npy')) # location * date
    
    p_dic = load_json(pred_path)
    
    p_npy = load_result(p_dic)
    
    assert a_npy.shape[0] == p_npy.shape[0], 'Missing some locations'
    assert a_npy.shape[1] == p_npy.shape[1], 'Missing some dates'
    assert a_npy.shape[2] == p_npy.shape[2], 'Missing some factors'
    
    '''
    for each location,
    1. get mse
    2. pub / pri
        a. pub/pri_mask = pub/pri * mask
        b. score += sqrt{(ans-pred)^2 * pub/pri_mask}
        c. count += sum(pub/pri_mask)
    '''
    
    score = 0
    pScore = 0
    
    count = 0
    pCount = 0
    
    for i in range(a_npy.shape[0]): # by location
        pub_mask = mask_npy[i] * pub_npy[i]
        prv_mask = mask_npy[i] * prv_npy[i]
        
        mse_matrix = (a_npy[i]-p_npy[i])**2
        
        score += np.sum(mse_matrix*pub_mask)
        pScore += np.sum(mse_matrix*prv_mask)
        
        count += np.sum(pub_mask)
        pCount += np.sum(prv_mask)

    score = (score / count)**.5
    pScore = (pScore / pCount)**.5

    assert score < 1e9, 'rmse is too big (greater than 1e9)'
    assert pScore < 1e9, 'rmse is too big (greater than 1e9)'

    
    return score, pScore

if __name__ == '__main__':
    answer = sys.argv[1]
    pred = sys.argv[2]
    
    answer = os.path.join(os.path.dirname(answer),'answer')

    try:
        import time
        start = time.time()
        score, pScore = mse(answer, pred)
        print(f'score={score},pScore={pScore}')
        print(f'Elapsed Time: {time.time() - start}')

    except Exception as e:
        print(f'evaluation exception error: {e}', file=sys.stderr)
        sys.exit()
