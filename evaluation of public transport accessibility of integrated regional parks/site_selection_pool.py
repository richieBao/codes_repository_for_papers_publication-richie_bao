# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:29:54 2021

@author: richie bao -Evaluation of public transport accessibility of integrated regional parks
"""

def Gini(k_v_list_s,args):
    import inequality
    import pandas as pd
    from copy import deepcopy
    
    existed_park_sp,col_name=args
    existed_park_sp_deepcopy=deepcopy(existed_park_sp)

    k,v=k_v_list_s
    existed_park_sp_deepcopy[k]=v
    update_time_cost=pd.concat([v[['stations','time_cost']].set_index(['stations']).rename(columns={'time_cost':k}) for k,v in existed_park_sp_deepcopy.items()],axis=1)
    update_tc_stats=update_time_cost.T.describe().T
    gini_updated=inequality.gini.Gini(update_tc_stats[col_name]).g
    # print(gini_updated)
    return (k,gini_updated)