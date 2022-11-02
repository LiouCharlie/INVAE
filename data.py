import scanpy as sc
import numpy as np
import torch
from scipy import sparse
def get_adata(anndata_path, pred_cell, targ_condition, pred_condition, cell_type_key, condition_type_key,  filt=False, isevalu=False):
    anndata = sc.read(anndata_path)
    
    train_conditions = [ targ_condition, pred_condition]
    anndata = anndata[(anndata.obs[condition_type_key].isin(train_conditions))]
    #if anndata.shape[1] > 2000:
    #  sc.pp.filter_genes_dispersion(anndata, n_top_genes=2000)
    
    if isevalu:
        pred_cell_indx  = np.unique(anndata.obs[cell_type_key])
        pred_condition_indx  = np.unique(anndata.obs[condition_type_key])
    else:
        pred_cell_indx  = np.unique(anndata.obs[cell_type_key])
        pred_cell_N = np.arange(len(pred_cell_indx))[pred_cell_indx == pred_cell][0]
        pred_condition_indx  = np.unique(anndata.obs[condition_type_key])
        targ_cond_N = np.arange(len(pred_condition_indx))[pred_condition_indx == targ_condition][0]
        pred_cond_N = np.arange(len(pred_condition_indx))[pred_condition_indx != targ_condition][0]
        
        mask = (anndata.obs[cell_type_key] == pred_cell) & (anndata.obs[condition_type_key] == targ_condition)
        anndata = anndata[mask!=True]
        #pred_cond_N = pred_condition_indx[pred_condition_indx != targ_condition][0]
    
    if (filt == True) and (isevalu==False):
        draw_num = int((len(anndata) / len(pred_cell_indx))//1)
        pred_num = ((anndata.obs[cell_type_key] == pred_cell)&(anndata.obs[condition_type_key]==pred_cond)).sum()
        draw_num = int((len(anndata)-pred_num) / (len(pred_cell_indx) - 1) // 1)
        mask_fil_cell = (anndata.obs[cell_type_key] == pred_cond)
        mask_1 = np.arange(len(anndata))[mask_fil_cell][np.random.permutation(mask_fil_cell.sum())[:draw_num]]    
        mask_2 = np.arange(len(anndata))[(anndata.obs[cell_type_key] != pred_cell)]
        mask = np.concatenate((mask_1,mask_2))
        anndata = anndata[mask]
    
    if sparse.issparse(anndata.X):
        adata = anndata.X.A
    else:
        adata = anndata.X
    
    lab_cell = np.zeros(len(adata),dtype = np.int)
    count = 0
    for i in pred_cell_indx:
        lab_cell[(anndata.obs[cell_type_key] == i).squeeze()] = int(count)
        count += 1

    lab_drug = np.zeros(len(adata),dtype = np.int)
    count = 0
    for i in pred_condition_indx:
        lab_drug[(anndata.obs[condition_type_key] == i).squeeze()] = int(count)
        count += 1

    adata = np.concatenate((adata, lab_cell.reshape([-1,1])), axis=1)
    adata = np.concatenate((adata, lab_drug.reshape([-1,1])), axis=1)
    fea_n = adata.shape[1] - 2
    adata = torch.tensor(adata,dtype=torch.float)
    if isevalu==False:
        return adata, pred_cell_N, targ_cond_N, pred_cond_N
    else:
        return adata, pred_cell_indx, pred_condition_indx

def test(prin):
    return prin