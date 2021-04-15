# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from pyro.nn import PyroModule 
import pyro
import pyro.distributions as dist 
import torch 

def GroundTruthModel():
    loc = 0 
    scale = 5
    R = pyro.sample("R", pyro.distributions.Normal(loc, scale))
    S = pyro.sample("S", pyro.distributions.Normal(loc, scale))
    K = pyro.sample("K", pyro.distributions.Normal(loc, scale))
    
    G = pyro.sample("G", dist.Normal(K + 4.0*R + 1.5*S, 0.1))
    L = pyro.sample("L", dist.Normal(K + 6.0*R + 0.5*S, 0.1))
    F = pyro.sample("F", dist.Normal(K + 3.0*R + 2.0*S, 0.1))
    

if __name__ == "__main__":
    """
    Race, Sex: R,S (protected)
    LSAT: L, GPA: G (features)
    FYA: F (outcome)
    Knowledge K (latent features)
    
    """
    
    trace_handler = pyro.poutine.trace(GroundTruthModel)
    samples = pd.DataFrame(columns=['R', 'S', 'K', 'G', 'L', 'F', 'p'])

    unaware_sample= []
    for i in range(1000):
        trace = trace_handler.get_trace()
        R = trace.nodes['R']['value']
        S = trace.nodes['S']['value']
        K = trace.nodes['K']['value']
        G = trace.nodes['G']['value']
        L = trace.nodes['L']['value']
        F = trace.nodes['F']['value']
        # get prob of each combination
        log_prob = trace.log_prob_sum()
        p = np.exp(log_prob)
        samples = samples.append({'R': R, 'S': S, 'K': K, 'G': G, 'L':L, 'F': F,'p':p}, ignore_index=True)
        unaware_sample.append(([G,L,F]))
    
    """Unaware Model: dont use the protected features"""
    unaware_sample = torch.tensor(unaware_sample)
    x_data, y_data = unaware_sample[:, :-1], unaware_sample[:, -1]
    
    linear_reg = PyroModule[torch.nn.Linear](2,1)       
    loss_func = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.Adam(linear_reg.parameters(), lr=0.05)
    num_iterations = 500
    
    for j in range(num_iterations):
        y_pred = linear_reg(x_data)
        loss = loss_func(y_pred, y_data)
        optim.zero_grad()
        loss.backward()
        optim.step()

    print("Learned parameters:")
    for name, param in linear_reg.named_parameters():
        print(name, param.data.numpy())
    
    """Full Model: use all the features"""
    
    
    """Inferring K
    G, L => K
    """
    G=samples.G
    L=samples.L
    K=[]
    for i in range(1000):
        conditioned = pyro.condition(GroundTruthModel, data={"G": G[i], "L": L[i]})
        posterior = pyro.infer.Importance(conditioned, num_samples=10).run()
        post_marginal = pyro.infer.EmpiricalMarginal(posterior, "K")
        post_samples = [post_marginal().item() for _ in range(10)]
        post_unique, post_counts = np.unique(post_samples, return_counts=True)
        mean = np.mean(post_samples)
        K.append(mean)
    
    
    
