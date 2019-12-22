# coding: utf-8
import numpy as np
import pandas as pd
import seaborn as sns

# MCMC Metropolis
class MCMC():
    
    def __init__(self, target, proposal_type="norm", burn_in=0.2, delta=1):
        
        self.proposal = None
        # Error Check
        if proposal_type not in ["norm", "unif"]:
            print("Error: Proposal distribution must be 'norm' or 'unif'")
            return
        elif burn_in < 0:
            print("Error: Parameter 'Burn in' must be over 0")
            return
        
        self.target   = target
        self.proposal = proposal_type
        self.burn_in  = burn_in
        self.delta    = delta
        self.samples  = []
        self.results  = []
        self.burn_ins = []
        
    def sample(self, curt_value):
        # 提案分布による次のステップの選択
        if self.proposal == "norm":
            return [cv + np.random.normal(0, self.delta) for cv in curt_value] 
        else:
            return [cv + np.random.uniform(-self.delta, self.delta) for cv in curt_value]
        
    def acceptance(self, curt_value, next_value):
        # 各位置における目標分布の確率密度の比
        P_curt = self.target(curt_value)  # 現在位置における目標分布の確率密度
        P_next = self.target(next_value)  # 候補位置における目標分布の確率密度
        # 比を計算
        r = P_next / P_curt
        if r > 1 or r > np.random.uniform(0, 1):
            return True
        return False
        
    def fit(self, N, curt_value):
        if self.proposal is None: return
        
        self.samples.append(curt_value)
        self.results  += [True]
        self.burn_ins += [False if 0 < self.burn_in else True]
        
        for n in range(N):
            next_value = self.sample(curt_value)
            self.samples.append(next_value)
            result     = self.acceptance(curt_value, next_value)
            self.results  += [result]
            curt_value     = next_value if result else curt_value
            self.burn_ins += [False if n+1 < N*self.burn_in else True]
                
    def convert2df(self, cols=None):
        s  = np.array(mcmc.samples)
        r  = np.array(mcmc.results).reshape(-1, 1)
        b  = np.array(mcmc.burn_ins).reshape(-1, 1)
        if cols is None:
            cols = ["x%s" % (i+1) for i in range(s.shape[1])]
        return pd.DataFrame(np.hstack([s,r,b]), columns=cols+["result","burn_in"])

def fn1(x, b=0.5):
    x1, x2 = x
    return np.exp(-0.5*(x1**2 - 2*b*x1*x2 + x2**2))

def fn2(x):
    x1, x2 = x
    return np.exp(-(5**2-(x1**2+x2**2))**2/200 + x2/20) * (6./5+np.sin(6*np.arctan2(x1,x2)))

mcmc = MCMC(fn2)
ndim = 2
curt_value = list(np.random.randint(-5, 5, ndim))

mcmc.fit(10000, curt_value)

df = mcmc.convert2df()
df_ext = df.query("burn_in==1 & result==1")
df_ext.shape

sns.jointplot(np.array(df_ext.x1), np.array(df_ext.x2))
