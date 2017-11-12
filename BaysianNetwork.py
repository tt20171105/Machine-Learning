# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 11:57:12 2017

@author: tt20171105
"""
import copy
import functools
import pandas as pd
from collections import OrderedDict

##############
#ベイジアンネットワーク
#
# 泥棒   地震
#  ↓     ↓
#  →→→ ←←←
#    ↓ ↓
#    警報
#     ↓
#    マリー
##############

#これは given の想定
#ベイジアンネットワークを作成する
def create_data():
    P_dorobo_1 = 0.001  #P(D=1)
    P_dorobo_0 = 0.999  #P(D=0)
    P_jisin_1  = 0.002  #P(J=1)
    P_jisin_0  = 0.998  #P(J=0)
    
    P_keiho_1_dorobo_1_jisin_1 = 0.99  #P(K=1 | D=1,J=1)
    P_keiho_0_dorobo_1_jisin_1 = 0.01  #P(K=0 | D=1,J=1)
    P_keiho_1_dorobo_1_jisin_0 = 0.96  #P(K=1 | D=1,J=0)
    P_keiho_0_dorobo_1_jisin_0 = 0.04  #P(K=0 | D=1,J=0)
    P_keiho_1_dorobo_0_jisin_1 = 0.28  #P(K=1 | D=0,J=1)
    P_keiho_0_dorobo_0_jisin_1 = 0.72  #P(K=0 | D=0,J=1)
    P_keiho_1_dorobo_0_jisin_0 = 0.01  #P(K=1 | D=0,J=0)
    P_keiho_0_dorobo_0_jisin_0 = 0.99  #P(K=0 | D=0,J=0)
    
    #P(K=1) = P(K=1 | D=1,J=1)P(D=1)P(J=1) + P(K=1 | D=1,J=0)P(D=1)P(J=0) + 
    #         P(K=1 | D=0,J=1)P(D=0)P(J=1) + P(K=1 | D=0,J=0)P(D=0)P(J=0)
    P_keiho_1 = P_keiho_1_dorobo_1_jisin_1 * P_dorobo_1 * P_jisin_1 + \
                P_keiho_1_dorobo_1_jisin_0 * P_dorobo_1 * P_jisin_0 + \
                P_keiho_1_dorobo_0_jisin_1 * P_dorobo_0 * P_jisin_1 + \
                P_keiho_1_dorobo_0_jisin_0 * P_dorobo_0 * P_jisin_0
    P_keiho_0 = 1 - P_keiho_1  #P(K=0)
    
    P_marry_1_keiho_1 = 0.88  #P(M=1 | K=1)
    P_marry_0_keiho_1 = 0.12  #P(M=0 | K=1)
    P_marry_1_keiho_0 = 0.32  #P(M=1 | K=0)
    P_marry_0_keiho_0 = 0.68  #P(M=0 | K=0)
    
    #P(M=1) = P(M=1 | K=1)P(K=1) + P(M=1 | K=0)P(K=0)
    P_marry_1 = P_marry_1_keiho_1 * P_keiho_1 + P_marry_1_keiho_0 * P_keiho_0
    P_marry_0 = 1 - P_marry_1  #P(M=0)
    
    df_priori_dorobo = pd.DataFrame([[1,P_dorobo_1],[0,P_dorobo_0]],columns=["value","p"])
    df_priori_jisin  = pd.DataFrame([[1,P_jisin_1], [0,P_jisin_0]], columns=["value","p"])
    df_conditional_keiho = pd.DataFrame([[1,1,1,P_keiho_1_dorobo_1_jisin_1],[0,1,1,P_keiho_0_dorobo_1_jisin_1],
                                         [1,1,0,P_keiho_1_dorobo_1_jisin_0],[0,1,0,P_keiho_0_dorobo_1_jisin_0],
                                         [1,0,1,P_keiho_1_dorobo_0_jisin_1],[0,0,1,P_keiho_0_dorobo_0_jisin_1],
                                         [1,0,0,P_keiho_1_dorobo_0_jisin_0],[0,0,0,P_keiho_0_dorobo_0_jisin_0]],columns=["keiho","dorobo","jisin","p"])
    df_posterior_keiho   = pd.DataFrame([[1,P_keiho_1],[0,P_keiho_0]],columns=["value","p"])
    df_conditional_marry = pd.DataFrame([[1,1,P_marry_1_keiho_1],[0,1,P_marry_0_keiho_1],
                                         [1,0,P_marry_1_keiho_0],[0,0,P_marry_0_keiho_0]],columns=["marry","keiho","p"])
    df_posterior_marry   = pd.DataFrame([[1,P_marry_1],[0,P_marry_0]],columns=["value","p"])
    
    dict_P = {"dorobo" : df_priori_dorobo,
              "jisin"  : df_priori_jisin,
              "keiho"  : df_posterior_keiho,
              "marry"  : df_posterior_marry}
    dict_conP = {"keiho" : df_conditional_keiho,
                 "marry" : df_conditional_marry}
    #cause のみのものは事前確率をもつ
    df_baysian_network = pd.DataFrame([["dorobo","keiho"],
                                       ["jisin","keiho"],
                                       ["keiho","marry"]],
                                       columns=["cause","result"])
    return dict_P, dict_conP, df_baysian_network

#親ノードを取得
def get_parent_node(_dict_parent, _dict_P, _df, _target, 
                    get_all = False, calc_num = 0, limit = 99):
    target_node = _df[_df["result"]==_target]["cause"]
    #なければリターン
    if (calc_num == limit) or (len(target_node)==0): return _dict_parent
    calc_num += 1
    #親ノードの値を辞書に入れる
    for curt_node in target_node:
        _dict_parent[curt_node] = list(_dict_P[curt_node]["value"].unique())
    if get_all:  #再帰的に探す
        for curt_node in target_node:
            _dict_parent = get_parent_node(_dict_parent, _dict_P, df_baysian_network, 
                                           curt_node, True, calc_num, limit)
    return _dict_parent

#子供ノードを取得
def get_children_node(_dict_children, _dict_P, _df, _target, 
                      get_all = False, calc_num = 0, limit = 99):
    target_node = _df[_df["cause"]==_target]["result"]
    #なければリターン
    if (calc_num == limit) or (len(target_node)==0): return _dict_children
    calc_num += 1
    #親ノードの値を辞書に入れる
    for curt_node in target_node:
        _dict_children[curt_node] = list(_dict_P[curt_node]["value"].unique())
    if get_all:  #再帰的に探す
        for curt_node in target_node:
            _dict_children = get_children_node(_dict_children, _dict_P, df_baysian_network, 
                                               curt_node, True, calc_num, limit)
    return _dict_children

#階層数を取得
def get_hierarchical_num(_df, _from, _to, hierarchical_num=1):
    #fromの親ノードを取得する
    parent_node = list(_df[_df["result"]==_from]["cause"])
    if len(parent_node) == 0:
        return 0  #なければリターン
    elif _to in parent_node:
        return hierarchical_num  #見つかればそこまでの階層数をリターン
    else:
        #次の親ノードを探す
        hierarchical_num += 1
        for curt_node in parent_node:
            #再帰的に探す
            hierarchical_num = get_hierarchical_num(_df, curt_node, _to, hierarchical_num)
            #見つかればそこまでの階層数をリターン
            if hierarchical_num != 0: return hierarchical_num
        return 0  #なければリターン

#ネットワークを展開する
def expand_network(_df, _from, _to, _list_expand):
    _list_expand.append(_from)  #ノードを追加
    #fromに紐づくノードを取得
    list_node = list(_df[_df.result==_from]["cause"])
    #取得したノードに存在すれば終了
    if _to in list_node:
        _list_expand.append(_to)
        return _list_expand
    #取得したノードに存在しない間は再帰的に探す
    for curt_node in list_node:
        _list_expand = expand_network(_df, curt_node, _to, _list_expand)
        #見つかれば終了
        if _to in _list_expand: return _list_expand
    return []  #なければリターン

#確率の算出
def get_probability(_df, _value):
    return _df[_df.value==_value]["p"].values[0]

#同時確率の算出
def get_joint_probability(_dict_P, _dict_conP, 
                          _evidence, _evidence_value, 
                          _target, _target_value, _target_probability, 
                          _list_conditional, _list_combinations):
    def exstract(_df, _col, _dict):
        if _col in _dict.keys():
            return _df[_df[_col] == _dict[_col]]
        return _df
    #エビデンス入力ノードから計算対象ノードまでのネットワークを取得する
    list_expand = []
    list_expand = expand_network(df_baysian_network, _evidence, _target, list_expand)
    
    joint_probability = []  #同時確率
    dict_value        = {}  #条件付き確率計算用のノードと値のdict
    for node_value in _list_combinations:
        conditional_probability = []
        for idx, node in enumerate(_list_conditional):
            dict_value[node] = node_value[idx]
            if node not in list_expand:
                probability = get_probability(_dict_P[node], node_value[idx])
                conditional_probability.append(probability)
        #条件付き確率表から必要な確率を取得する
        for curt_node in list_expand:
            if curt_node not in _dict_conP.keys(): continue
            df = _dict_conP[curt_node]
            if curt_node == _evidence: df = df[df[_evidence]==_evidence_value]
            for col in df:
                df = exstract(df, col, dict_value)
                if col == _target: df = df[df[_target]==_target_value]
            p = df["p"].values[0]
            conditional_probability.append(p)
        #周辺化
        conditional_probability  = functools.reduce(lambda x,y: x*y, conditional_probability)
        conditional_probability *= _target_probability
        joint_probability.append(conditional_probability)
    return sum(joint_probability)

#エビデンス情報をもとに親ノードを再計算する
def parent_node_recalculation(_evidence, _evidence_value, _dict_target,
                              _dict_P_for_calc, _dict_P_new):
    node_num = 0
    for target, target_value in _dict_target.items():
        for curt_value in target_value:
            #計算対象ノードの確率を求める
            target_probability   = get_probability(_dict_P_new[target], curt_value)
            #エビデンス入力ノードの確率を求める
            evidence_probability = get_probability(_dict_P_new[_evidence], _evidence_value)
            
            #エビデンス入力ノードと計算対象ノードとの階層構造を取得する
            dict_parent = OrderedDict()
            dict_parent = get_parent_node(dict_parent, dict_P, df_baysian_network, 
                                          _evidence, True, limit = node_num + 1)
            
            #同時確率を求めるための他のノードとの組み合わせを取得する
            list_conditional  = []
            list_combinations = []
            for another, another_value in dict_parent.items():
                if target == another: continue
                list_conditional.append(another)
                if list_combinations == []:
                    #親ノードが２つ
                    list_combinations = [[x] for x in dict_parent[another]]
                else:
                    #親ノードが３つ以上
                    list_combinations = [x + [y] for x in list_combinations for y in dict_parent[another]]
            #同時確率を求める
            if list_conditional == []:
                df_conP = dict_conP[_evidence]
                #親ノードが１つ
                joint_probability = df_conP[(df_conP[target]==curt_value)& 
                                            (df_conP[_evidence]==_evidence_value)]["p"].values[0] * \
                                    get_probability(_dict_P_new[target], curt_value)
            else:
                #親ノードが２つ以上
                joint_probability = get_joint_probability(_dict_P_new, dict_conP, 
                                                          _evidence, _evidence_value,
                                                          target, curt_value, target_probability, 
                                                          list_conditional, list_combinations)
            #更新後の値を計算する
            new_probability = joint_probability / evidence_probability
            #計算結果を一時退避
            _dict_P_for_calc[target].loc[_dict_P_for_calc[target].value==curt_value,"p"] = new_probability
    
        #階層構造が変わればインクリメント
        if node_num != get_hierarchical_num(df_baysian_network, _evidence, target):
            node_num += 1
    
    #計算結果を格納し、エビデンス情報を反映する
    _dict_P_new = copy.deepcopy(_dict_P_for_calc)
    _dict_P_new[_evidence].loc[_dict_P_new[_evidence].value==_evidence_value,"p"] = 1
    _dict_P_new[_evidence].loc[_dict_P_new[_evidence].value!=_evidence_value,"p"] = 0
    return _dict_P_new

#######################
dict_P, dict_conP, df_baysian_network = create_data()

#エビデンスは keiho か marry しか対応してない
evidence       = "keiho"
evidence_value = 1

dict_P_for_calc = copy.deepcopy(dict_P)  #計算途中の結果を保存
dict_P_new      = copy.deepcopy(dict_P)  #計算後の結果を保存

dict_target = OrderedDict()
dict_target = get_parent_node(dict_target, dict_P, df_baysian_network, evidence, True)
dict_P_new  = parent_node_recalculation(evidence, evidence_value, dict_target,
                                        dict_P_for_calc, dict_P_new)

for key, value in dict_P.items():
    for key_new, value_new in dict_P_new.items():
        if key != key_new: continue
        print(key)
        df_merged = pd.merge(value, value_new, on="value", suffixes=("_before","_after"))
        print(df_merged)
        print("")

#P(D=1 | K=1) = P(D=1,K=1) / P(K=1)
#P(D=1,K=1)   = P(K=1 | D=1,J=1)P(D=1)P(J=1) + P(K=1 | D=1,J=0)P(D=1)P(J=0)

#P(J=0 | K=1) = P(J=0,K=1) / P(K=1)
#P(J=0,K=1)   = P(K=1 | D=1,J=0)P(D=1)P(J=0) + P(K=1 | D=0,J=0)P(D=0)P(J=0)

#P(J=1 | K=1) = P(J=1,K=1) / P(K=1)
#P(J=1,K=1)   = P(K=1 | J=1)P(J=1)

#P(K=1 | M=1) = P(K=1,M=1) / P(M=1)
#P(K=1,M=1)   = P(M=1 | K=1)P(K=1)

#P(M＝1|K＝1)P(K＝1|D＝1,J＝1)P(D＝1)P(J＝1) + 
#P(M＝1|K＝1)P(K＝1|D＝1,J＝0)P(D＝1)P(J＝0) + 
#P(M＝1|K＝0)P(K＝0|D＝1,J＝1)P(D＝1)P(J＝1) + 
#P(M＝1|K＝0)P(K＝0|D＝1,J＝0)P(D＝1)P(J＝0)
