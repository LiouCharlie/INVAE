import anndata
from model import *
from utils import *
from data import *

class INVAE(nn.Module):
    
    def __init__(self, fea_n, bat_N=512, C_lat_num=30, S_lat_num=30, meta_N=5):
        super(INVAE, self).__init__()
        
        self.fea_n = fea_n
        self.cell_lat_num = C_lat_num
        self.drug_lat_num = S_lat_num
        self.bat_N = bat_N
        self.meta_N = meta_N
        self.p_layer = P_Layer((S_lat_num + 1)).cuda()
        self.encoder = MLPEncoder((S_lat_num + C_lat_num)*2, fea_n=fea_n).cuda()
        self.decoder_C = MLPDecoder_C((C_lat_num + 1), fea_n=fea_n).cuda()
        self.decoder_S = MLPDecoder_S((S_lat_num + 1), fea_n=fea_n).cuda()
        self.prior_params = torch.zeros(bat_N,(C_lat_num + S_lat_num), 2).cuda()
        self.dist = Normal()

    def train(self, adata, pred_cell_N, pred_cond_N, iter_N):
        tot_num = len(adata)
        prior_params = self.prior_params
        x = adata.cuda() 
        x_LC = adata[:,-2].cuda()
        x_LD = adata[:,-1].cuda()

        z_params_o = self.encoder.forward(x).view(x.size(0), (self.cell_lat_num + self.drug_lat_num), 2)
        z_params_o = torch.clamp(z_params_o, min=-30,max=30)
        z_params = self.dist.sample(params=z_params_o).cuda()


        z_com = z_params[:,:(self.cell_lat_num)]
        z_spe = z_params[:,(self.cell_lat_num):(self.cell_lat_num + self.drug_lat_num)]
        impute_D = x_LD.reshape([-1,1])

        impute_C = x_LC.reshape([-1,1])
        #------------------ #tf learning
        mask_pred = x_LC == pred_cell_N  
        mask_train = x_LC != pred_cell_N
        if mask_pred.sum() > 0:
            inp_spe = torch.cat((z_spe,impute_D),1)[mask_pred]
            inp_con = torch.cat((z_com,impute_C),1)[mask_pred]

            z_proj_N = self.p_layer.forward(inp_spe)
            x_params_spe_N = self.decoder_S.forward(z_proj_N)
            x_params_spe_N = x_params_spe_N.view(mask_pred.sum(), self.fea_n)
            x_params_com_N = self.decoder_C.forward(inp_con).view(mask_pred.sum(), self.fea_n)
            x_params_N = x_params_spe_N + x_params_com_N 
            original_recovery_N = ((x[mask_pred,:-2] - x_params_N)**2).sum(1)
        
        z_proj_N = self.p_layer.forward(torch.cat((z_spe,impute_D),1))
        x_params_spe = self.decoder_S.forward(z_proj_N)
        x_params_spe = x_params_spe.view(z_params.size(0), self.fea_n)
        x_params_com = self.decoder_C.forward(torch.cat((z_com,impute_C),1)).view(z_com.size(0), self.fea_n)
        x_params = x_params_com + x_params_spe 
        original_recovery = ((x[mask_train,:-2] - x_params[mask_train])**2).sum(1)
        #----------------  #projection layer loss calculation

        impute_D_ = x_LD.reshape([-1,1]).clone()
        impute_D_[x_LD.reshape([-1,1]) == torch.unique(x_LD.reshape([-1,1]))[1]] = torch.unique(x_LD.reshape([-1,1]))[0]
        impute_D_[x_LD.reshape([-1,1]) == torch.unique(x_LD.reshape([-1,1]))[0]] = torch.unique(x_LD.reshape([-1,1]))[1]
        z_proj_rev = self.p_layer.forward(torch.cat((z_spe,impute_D_),1))

        mask_d_0 = ( x_LD.reshape([-1,1]) == torch.unique(x_LD.reshape([-1,1]))[0]).squeeze()
        mask_d_1 = ( x_LD.reshape([-1,1]) == torch.unique(x_LD.reshape([-1,1]))[1]).squeeze()

        z_proj_1 = torch.cat((z_proj_N[mask_d_0],z_proj_rev[mask_d_1]),0)
        proj_1_C = torch.cat((x_LC[mask_d_0],x_LC[mask_d_1]),0)
        proj_1_D = torch.cat((torch.zeros(mask_d_0.sum()),torch.ones(mask_d_1.sum())),0).cuda()
        L_projection = anova_loss(z_proj_1,proj_1_C,proj_1_D).mean()

        z_proj_2 = torch.cat((z_proj_N[mask_d_1],z_proj_rev[mask_d_0]),0)
        proj_2_C = torch.cat((x_LC[mask_d_1],x_LC[mask_d_0]),0)
        proj_2_D = torch.cat((torch.zeros(mask_d_1.sum()),torch.ones(mask_d_0.sum())),0).cuda()
        L_projection += anova_loss(z_proj_2,proj_2_C,proj_2_D).mean()
        #=======

        logpz = self.dist.log_density(z_params, params=prior_params).view(self.bat_N, -1).sum(1)
        _logqz = self.dist.log_density(
                z_params.view(self.bat_N, 1, (self.cell_lat_num + self.drug_lat_num)),
                z_params_o.view(1, self.bat_N, (self.cell_lat_num + self.drug_lat_num), 2)
            )
        logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(self.bat_N)).sum(1)
        logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(self.bat_N))



        L_TCV_KL =  (2*(logqz - logqz_prodmarginals)) + (logqz_prodmarginals - logpz)


        #-------------------
        L_INV_1 = anova_loss(z_com,x_LC,x_LD).mean()
        L_INV_2 = anova_loss(z_spe[x_LD==pred_cond_N],x_LD[x_LD==pred_cond_N],x_LC[x_LD==pred_cond_N]).mean()
        if torch.isinf(L_projection):
            L_INV = L_INV_1 + L_INV_2
        else:
            L_INV = L_INV_1 + L_INV_2 + L_projection
        #-----

        if mask_pred.sum() > 0:
            if iter_N % self.meta_N == 0:
                L_tot = ( L_INV + L_TCV_KL ) + (original_recovery_N.sum() + original_recovery.sum())
            elif iter_N % self.meta_N >= 1:
                L_tot = ( L_INV + L_TCV_KL ) + original_recovery.sum()
        else:
            L_tot = ( L_INV + L_TCV_KL ) + original_recovery.sum()
        
        return L_tot
                              


def main(ann_path, pred_cell, targ_condition, pred_condition,epoch_N, bat_N, cell_type_key, condition_type_key, real_mean_):
        
    adata, pred_cell_N, targ_cond_N, pred_cond_N = get_adata(anndata_path = ann_path,  pred_cell = pred_cell, targ_condition = targ_condition, pred_condition = pred_condition, cell_type_key=cell_type_key, condition_type_key=condition_type_key)
    
    fea_n = adata.shape[1] - 2
    
    iter_N = 0
    learning_rate = 1*(10)**-3
    model = INVAE(fea_n = fea_n, bat_N = bat_N)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dist = Normal()
    eps = 1*10**(-8)
    dist = Normal().cuda()
    
    for i in range(epoch_N):
        shuffle = np.random.permutation(len(adata))
        adata = adata[shuffle]
        #print(i)
        #if i != 0:
            #print(L_tot.mean(),iter_N)
        for j in range(len(adata)//bat_N):
            optimizer.zero_grad()
            bat_adata = adata[(j*bat_N):((j+1)*bat_N)]
            L_tot = model.train(bat_adata, pred_cell_N, pred_cond_N, iter_N)
            
            L_tot.mean().backward()
            optimizer.step()
            iter_N += 1
            
        if (i%5) == 0:
            x_ = adata[adata[:,-2]==pred_cell_N]
            x_params_N_T = np.zeros([x_.shape[0],x_.shape[1]-2])
            
            for i_t in range(len(x_)//1000 + 1):
                x = x_[(i_t*1000):((i_t+1)*1000)]
                z_params_o = model.encoder.forward(x.cuda()).view(x.size(0), (60), 2)
                z_params_o = torch.clamp(z_params_o, min=-30,max=30)
                z_params = Normal().sample(params=z_params_o).cuda()
                z_com = z_params[:,:(30)]
                z_spe = z_params[:,(30):(60)]
                impute_D = x[:,-1].reshape([-1,1]).cuda() 
                impute_D[:] = targ_cond_N
                impute_C = x[:,-2].reshape([-1,1]).cuda()

                inp_spe = torch.cat((z_spe,impute_D),1)
                inp_con = torch.cat((z_com,impute_C),1)

                z_proj_N = model.p_layer.forward(inp_spe)
                x_params_spe_N = model.decoder_S.forward(z_proj_N)
                x_params_spe_N = x_params_spe_N.view(len(inp_con), fea_n)
                x_params_com_N = model.decoder_C.forward(inp_con).view(len(inp_con), fea_n)
                x_params_N = x_params_spe_N + x_params_com_N 
                x_params_N_T[(i_t*1000):((i_t)*1000 + len(x_params_N))] = x_params_N.cpu().detach().numpy()
            pred_mean = np.mean(x_params_N_T, axis=0)
            print('===============eva==========================')
            print('r-squared:',(1 - (((pred_mean-real_mean_)**2).sum() / ((real_mean_-real_mean_.mean())**2).sum())))
            x_params_N_T = np.zeros([x_.shape[0],x_.shape[1]-2])
            pred_mean = np.mean(x_params_N_T, axis=0)
            #print((1 - (((pred_mean-real_mean_)**2).sum() / ((real_mean_-real_mean_.mean())**2).sum())),'hand_craft')
            print('')
   
    return model
            
                                         
                                   
