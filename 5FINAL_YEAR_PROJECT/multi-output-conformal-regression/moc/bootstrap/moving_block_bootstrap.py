import numpy as np
import math
import torch
import copy
from torch.utils.data.dataset import Subset
from moc.datamodules import bootstrap_load_datamodule
from moc.datamodules.base_datamodule import ScaledDataset
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class MovingBlockBootstrap:
    def __init__(self, block_size,bootstrap_trials,model,rc,trainer,config_device):
        self.block_size = block_size
        self.bootstrap_trials = bootstrap_trials
        self.model = model 
        self.rc = rc 
        self.emptytrainer = trainer
        self.datamodule = bootstrap_load_datamodule(rc)
        self.config_device = config_device
        self.train_data = next(iter(self.datamodule.train_dataloader()))
        self.scaler_x = self.datamodule.data_train.scaler_x
        self.scaler_y = self.datamodule.data_train.scaler_y

    def sample(self):
        train_data = self.train_data
        data_size = train_data[1].shape[0]
        sample_length = (data_size)/self.block_size
        indices = np.random.randint(0,data_size-self.block_size, size=math.ceil(sample_length))
        indices = [int(i) for i in indices]
        train_X = train_data[0][indices[0]:indices[0]+self.block_size]
        train_Y = train_data[1][indices[0]:indices[0]+self.block_size]
        for i in range(1,len(indices)):
            x = train_data[0][indices[i]:indices[i]+self.block_size]
            y = train_data[1][indices[i]:indices[i]+self.block_size]
            train_X = torch.cat((train_X,x),dim=0)
            train_Y = torch.cat((train_Y,y),dim=0)
        if data_size != math.ceil(sample_length)*self.block_size:
            train_X = train_X[:data_size - math.ceil(sample_length)*self.block_size]
            train_Y = train_Y[:data_size - math.ceil(sample_length)*self.block_size]
        indices = [i for i in range(0,len(train_X))]
        tensor_data = TensorDataset(train_X,train_Y)
        (self.datamodule.data_train,) = [Subset(tensor_data,indices)]
    
    def bootstrap_sample(self,alpha,region_sizes,marginal_coverage,ssc,ewcf,ebscf,kmeans):
        #need to fit model and then calculate metrics 
        val_batch = next(iter(self.datamodule.val_dataloader()))
        val_x, val_y = val_batch
        region_sizes_total = np.zeros(val_x.shape[0])
        marginal_coverage_total = []
        ssc_total = [] 
        ewcf_total = []
        ebscf_total = []
        kmeans_total = []
        for i in range(self.bootstrap_trials):
            self.sample()
            empty_trainer = copy.deepcopy(self.emptytrainer)
            model = copy.deepcopy(self.model)
            empty_trainer.fit(model,self.datamodule)
            model.to(self.config_device)
            if region_sizes:
                region_sizes_total+=self.bs_region_size(alpha,val_x,model,100)
            if marginal_coverage:
                marginal_coverage_total.append(self.bs_marginal_coverage(alpha,val_x,val_y,model,100))
            if ssc:
                ssc_total+=[self.bs_size_stratified_coverage(alpha,val_x,val_y,model,8,100)]
            if ewcf:
                ewcf_total.append(self.bs_continuous_fsc(model,alpha,val_x,val_y,strat_tech="Equal_width"))
            if ebscf:
                ebscf_total.append(self.bs_continuous_fsc(model,alpha,val_x,val_y,strat_tech="Equal_bin_size"))
            if kmeans:
                x_train, _ = next(iter(self.datamodule.train_dataloader()))
                kmeans_total.append(self.bs_continuous_fsc(model,alpha,val_x,val_y,strat_tech="kmeans",x_train=x_train))
            
        marginal_coverage_total = sum(marginal_coverage_total)/self.bootstrap_trials
        region_sizes_total/=self.bootstrap_trials
        self.plt_region_size(region_sizes_total,10)
        ssc_total = [sum(x)/self.bootstrap_trials for x in zip(*ssc_total)]

        w_total = np.zeros((len(ewcf_total[0]), len(ewcf_total[0][0])))
        for i in range(len(ewcf_total)):
            w_total += np.array(ewcf_total[i])
        ewcf_total = w_total/self.bootstrap_trials

        b_total = np.zeros((len(ebscf_total[0]), len(ebscf_total[0][0])))
        for i in range(len(ebscf_total)):
            b_total += np.array(ebscf_total[i])
        ebscf_total = b_total/self.bootstrap_trials
        kmeans_total = [sum(x)/self.bootstrap_trials for x in zip(*kmeans_total)]
        w_bins = ["Width bin "+str(i+1) for i in range(len(ewcf_total[0]))]
        b_bins = ["Bin "+str(i+1) for i in range(len(ebscf_total[0]))]
        k_bins = ["Bin "+str(i+1) for i in range(len(kmeans_total))]
        print(ssc_total)
        ssc_bins = ["Bin "+str(i+1) for i in range(len(ssc_total))]

        for i in range(len(ewcf_total)):
            self.stratified_plot(w_bins, ewcf_total[i],1,w_bins,"Width Bin Classes", "Coverage","Equal Width Feature Stratified Conditional Coverage of Feature "+str(i+1))
        for i in range(len(ebscf_total)):
            self.stratified_plot(b_bins,ebscf_total[i],1,b_bins,"Bin Classes","Coverage","Equal Bin Size Feature Stratified Conditional Coverage of Feature "+str(i+1))

        self.stratified_plot(k_bins,kmeans_total,1,k_bins,"Bin Classes","Coverage","Kmeans Feature Stratified Conditional Coverage")
        self.stratified_plot(ssc_bins, ssc_total,1,ssc_bins,"Bin Classes","Coverage","Size stratified Coverage")
        print("marginal_coverage",marginal_coverage_total)
    
       
        
       
       
       
    def bs_region_size(self,alpha,x,model,n_sample,cache={}):
        dist = model.predict(x)
        samples = dist.sample((n_sample,))
        ql,qh = torch.quantile(samples, torch.tensor([alpha / 2, 1 - alpha / 2], device=x.device), dim=0)
        prediction_interval = torch.maximum(qh - ql, torch.tensor(1e-50, device=x.device))
        region_sizes = prediction_interval.prod(dim=-1) 
        region_sizes_np = region_sizes.numpy()
        return region_sizes_np
    
    def bs_marginal_coverage(self,alpha,x,y,model,n_sample,cache={}):
        dist = model.predict(x)
        samples = dist.sample((n_sample,))
        ql,qh = torch.quantile(samples, torch.tensor([alpha / 2, 1 - alpha / 2], device=x.device), dim=0)
        counter = 0
        for i in range(y.size()[0]):
            lower_check = y[i]>=ql[i]
            higher_check = y[i]<=qh[i]
            if lower_check.all().item() and higher_check.all().item():
                counter+=1
        return counter/y.size()[0]
    
    def bs_size_stratified_coverage(self,alpha,x,y,model,bins,n_sample,cache ={}):
        dist = model.predict(x)
        samples = dist.sample((n_sample,))
        ql,qh = torch.quantile(samples, torch.tensor([alpha / 2, 1 - alpha / 2], device=x.device), dim=0)
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
        return coverage
    
    def bs_equal_width_continuous_fsc(self,x,y,ql,qh,bins=5):
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
            new_coverage = []
            for (a,b) in zip(counter,total):
                if b!= 0:
                    new_coverage.append(float(f"{(a/b):.3g}"))
                else:
                    new_coverage.append(0)
            coverage = [float(f"{(x/y):.3g}") for x, y in zip(counter, total) if y!=0 ]
            coverages.append(new_coverage) 
        return coverages  

    def bs_equal_bin_size_continuous_fsc(self,x,y,ql,qh,bins=5):
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
            coverages.append(coverage)  
        return coverages  

    def bs_kmeans_continuous_fsc(self,x,y,ql,qh,x_train,bins=5):
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
        return coverage   

    def bs_continuous_fsc(self,model,alpha,x,y,cols=[],strat_tech="Equal_width",bins=5,cache={},x_train=[],n_sample=100):
        coverages = []
        x2 = x
        if cols!=[]:
            x2 = x[:,cols]
        dist = model.predict(x2)
        samples = dist.sample((n_sample,))
        ql,qh = torch.quantile(samples, torch.tensor([alpha / 2, 1 - alpha / 2], device=x.device), dim=0)
        if strat_tech == "Equal_width":
            coverages = self.bs_equal_width_continuous_fsc(x2,y,ql,qh,bins=5)
        elif strat_tech == "Equal_bin_size":
            coverages = self.bs_equal_bin_size_continuous_fsc(x2,y,ql,qh,bins=bins)
        elif strat_tech == "kmeans":
            coverages = self.bs_kmeans_continuous_fsc(x,y,ql,qh,bins=bins,x_train=x_train)
        else:
            assert False, strat_tech+ " should be Equal_width, Equal_bin_size or kmeans"
        return coverages
                    
    def stratified_plot(self,bin, coverage,width,intervals,x_label,y_label,title):
        plt.figure(figsize=(8, 5))
        plt.bar(bin, coverage, color='skyblue',edgecolor ='black', width=width)
        plt.xticks(intervals, bin)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()


    def plt_region_size(self,region_sizes_np,bins):
        min_val, max_val = region_sizes_np.min(), region_sizes_np.max()
        x_axis = np.linspace(min_val, max_val, num=bins + 1)
        plt.hist(region_sizes_np, bins=x_axis,color='skyblue',edgecolor='black')
        plt.xlabel("Region Sizes")
        plt.ylabel("Frequency")
        plt.show()