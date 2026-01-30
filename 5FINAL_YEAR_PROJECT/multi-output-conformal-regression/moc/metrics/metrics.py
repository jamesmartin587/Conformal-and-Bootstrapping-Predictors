import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def compute_marginal_coverage(conformalizer, alpha, x, y, cache = {}):
    return "alpha: "+str(alpha)+"\nmarginal coverage: "+str(conformalizer.is_in_region(x, y, alpha, cache).float().mean().item())

def empirical_coverage(conformalizer,alpha,x,y,cache={}):
    ql,qh = conformalizer.get_region_bounds(x,alpha,cache)
    q = conformalizer.get_q(alpha)
    ql,qh = ql-q, qh+q 
    counter = 0
    for i in range(y.size()[0]):
        lower_check = y[i]>=ql[i]
        higher_check = y[i]<=qh[i]
        if lower_check.all().item() and higher_check.all().item():
            counter+=1
    return counter/y.size()[0]

def stratified_plot(bin, coverage,width,intervals,x_label,y_label,title):
    plt.figure(figsize=(8, 5))
    plt.bar(bin, coverage, color='skyblue',edgecolor ='black', width=width)
    plt.xticks(bin, intervals)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def equal_width_continuous_fsc(x,y,ql,qh,bins=5):
    coverages = []
    for i in range(x.size()[1]):
        counter = [0]*bins
        total = [0]*bins
        width = (torch.max(x[:,i]) - torch.min(x[:,i]))/bins
        for j in range(y.size()[0]):
            lower_check = y[j]>=ql[j]
            higher_check = y[j]<=qh[j]
            index = int((x[j,i]-torch.min(x[:,i]))//width)
            if x[j,i] == torch.max(x[:,i]):
                index -=1
            if lower_check.all().item() and higher_check.all().item():
                counter[index]+=1
            total[index]+=1
        bin = np.linspace(torch.min(x[:,i]), torch.max(x[:,i]), num=bins + 1)
        bin = bin.tolist()
        intervals = [f"{bin[i]:.3g} to {bin[i+1]:.3g}" for i in range(len(bin) - 1)]
        bin.pop()
        zero_indices = [i for i,val in enumerate(total) if val == 0]
        total = [item for i, item in enumerate(total) if i not in zero_indices]
        counter = [item for i, item in enumerate(counter) if i not in zero_indices]
        bin = [item for i,item in enumerate(bin) if i not in zero_indices]
        intervals = [item for i,item in enumerate(intervals) if i not in zero_indices] 
        coverage = [float(f"{(x/y):.3g}") for x, y in zip(counter, total)]
        stratified_plot(bin,coverage,width,intervals,'Bin widths','Coverage','Feature Stratified Coverage with Equal Width Intervals')
        coverages.append(sum(coverage)/len(coverage)) 
    return coverages  

def equal_bin_size_continuous_fsc(x,y,ql,qh,bins=5):
    coverages = []
    for i in range(x.size()[1]):
        counter = [0]*bins
        total = [0]*bins
        sorted_x = torch.sort(x[:,i])
        for j in range(y.size()[0]):
            lower_check = y[j]>=ql[j]
            higher_check = y[j]<=qh[j]
            index = sorted_x[1][j]
            bin_index = int(index//(y.size()[0]/bins))
            if lower_check.all().item() and higher_check.all().item():
                counter[bin_index]+=1
            total[bin_index]+=1
        coverage = [float(f"{(x/y):.3g}") for x, y in zip(counter, total)]
        intervals = []
        for b in range(bins):
            intervals.append(str(b*total[b])+" to "+str((b+1)*total[b]-1))
        bin = [d*y.size()[0]/bins for d in range(bins)]
        stratified_plot(bin,coverage,y.size()[0]/bins,intervals,"Bin Index","Coverage","Feature Stratified Coverage with Equal Bin Sizes")
        coverages.append(sum(coverage)/len(coverage))  
    return coverages  

def kmeans_continuous_fsc(x,y,ql,qh,bins=5,x_train=[]):
    kmeans = KMeans(n_clusters=bins,random_state=317)
    kmeans.fit(x_train)
    val_clusters = kmeans.predict(x)
    counter = [0]*bins
    total = [0]*bins
    for j in range(y.size()[0]):
        lower_check = y[j]>=ql[j]
        higher_check = y[j]<=qh[j]
        bin_index = val_clusters[j]
        if lower_check.all().item() and higher_check.all().item():
            counter[bin_index]+=1
        total[bin_index]+=1
    zero_indices = [i for i,val in enumerate(total) if val == 0]
    total = [item for i, item in enumerate(total) if i not in zero_indices]
    counter = [item for i, item in enumerate(counter) if i not in zero_indices]
    bins-=len(zero_indices)
    coverage = [float(f"{(x/y):.3g}") for x, y in zip(counter, total)]
    intervals = [f"Cluster {a+1}" for a in range(bins)]
    bin = [d*y.size()[0]/bins for d in range(bins)]
    stratified_plot(bin,coverage,y.size()[0]/bins,intervals,"Bin Index","Coverage","Feature Stratified Coverage with Kmeans")
    return coverage   

def continuous_fsc(conformalizer,alpha,x,y,cols=[],strat_tech="Equal_width",bins=5,cache={},x_train=[]):
    coverages = []
    x2 = x
    if cols!=[]:
        x2 = x[:,cols]
    ql,qh = conformalizer.get_region_bounds(x,alpha,cache)
    q = conformalizer.get_q(alpha)
    ql,qh = ql-q, qh+q 
    if strat_tech == "Equal_width":
        coverages = equal_width_continuous_fsc(x2,y,ql,qh,bins=bins)
    elif strat_tech == "Equal_bin_size":
        coverages = equal_bin_size_continuous_fsc(x2,y,ql,qh,bins=bins)
    elif strat_tech == "kmeans":
        coverages = kmeans_continuous_fsc(x,y,ql,qh,bins=bins,x_train=x_train)
    else:
        assert False, strat_tech+ " should be Equal_width, Equal_bin_size or kmeans"
    return coverages

def region_size(conformalizer,alpha,x,bins,cache={},plot=False):
    ql, qh = conformalizer.get_region_bounds(x, alpha, cache)
    q = conformalizer.get_q(alpha)
    ql, qh = ql - q, qh + q
    prediction_interval = torch.maximum(qh - ql, torch.tensor(1e-50, device=x.device))
    region_sizes = prediction_interval.prod(dim=-1) 
    region_sizes_np = region_sizes.numpy()
    if plot: 
        min_val, max_val = region_sizes_np.min(), region_sizes_np.max()
        x_axis = np.linspace(min_val, max_val, num=bins + 1)
        plt.hist(region_sizes_np, bins=x_axis,color='skyblue',edgecolor='black')
        plt.xlabel("Region Sizes")
        plt.ylabel("Frequency")
        plt.show()
    return region_sizes.mean()

def size_stratified_coverage(conformalizer,alpha,x,y,bins,cache ={}):
    ql, qh = conformalizer.get_region_bounds(x, alpha, cache)
    q = conformalizer.get_q(alpha)
    ql, qh = ql - q, qh + q
    widths = torch.prod(qh-ql,dim=-1)
    sorted_widths_indices = np.argsort(widths)
    counter = [0]*bins
    total = [0]*bins
    bin_sizes = y.size()[0]/bins
    for i in range(y.size()[0]):
        lower_check = y[i]>=ql[i]
        higher_check = y[i]<=qh[i]
        bin_index = int(sorted_widths_indices[i]//bin_sizes)
        if lower_check.all().item() and higher_check.all().item():
            counter[bin_index]+=1
        total[bin_index]+=1  
    coverage = [float(f"{(x/y):.3g}") for x, y in zip(counter, total)]
    intervals = []
    for b in range(bins):
        intervals.append(str(sum(total[:b]))+" to "+str(sum(total[:b+1])-1))
    bin = [d*y.size()[0]/bins for d in range(bins)]
    stratified_plot(bin,coverage,y.size()[0]/bins,intervals,"Bin Index Intervals","Coverage","Size Stratified Coverage")    
    return min(coverage)

def categorical_fsc(conformalizer,alpha,x,y,cols=[],cache={}):
    ql,qh = conformalizer.get_region_bounds(x,alpha,cache)
    q = conformalizer.get_q(alpha)
    ql,qh = ql-q, qh+q 
    coverages = []
    x2 = x
    if cols != []:
        x2 = x[:,cols]
    for i in range(x2.size()[1]):
        unique_x = torch.unique(x2[:,i])
        counter = [0]*unique_x.size()[0]
        total = [0]*unique_x.size()[0]
        for j in range(y.size()[0]):
            lower_check = y[j]>=ql[j]
            higher_check = y[j]<=qh[j]
            bin_index = (unique_x == x2[j,i]).nonzero(as_tuple=True)[0]
            if lower_check.all().item() and higher_check.all().item():
                counter[bin_index]+=1
            total[bin_index]+=1 
        coverage = [float(f"{(x/y):.3g}") for x, y in zip(counter, total)]
        bin = [str(d*unique_x.size()[0]) for d in range(unique_x.size()[0])]
        labels = [str(f"{x:.3g}") for x in unique_x]
        stratified_plot(bin,coverage,1,labels,"Bin Classes","Coverage","Classification Feature Stratified Conditional Coverage")  
        coverages.append(min(coverage))
    return coverages

def fsc(conformalizer,alpha,x,y,categ_conti,strat_tech="Equal_width",bins=5,cache={},x_train=[]):
    coverages = []
    if len(categ_conti) != x.size()[1]:
        assert False, "categ_conti needs to state 0 for Categorical or 1 for Continuous to indicate each Feature."
    if strat_tech == "kmeans":
        coverages = continuous_fsc(conformalizer,alpha,x,y,strat_tech=strat_tech,bins=bins,cache=cache,x_train=x_train)
    elif strat_tech == "Equal_width" or strat_tech == "Equal_bin_size":
        for i in range(len(categ_conti)):
            if categ_conti[i] == 0:
                coverage = categorical_fsc(conformalizer,alpha,x,y,cols=[i],cache=cache)
            elif categ_conti[i] == 1:
                coverage = continuous_fsc(conformalizer,alpha,x,y,cols=[i],strat_tech=strat_tech,bins=bins,cache=cache,x_train=x_train)
            else:
                assert False, "categ_conti must be either 0 for Categorical or 1 for Continuous."
        coverages.append(coverage)
    else:
        assert False, "For continuous fsc the strat tech needs to be either 'Equal_width', 'Equal_bin_size' or 'kmeans'"    
    return coverages