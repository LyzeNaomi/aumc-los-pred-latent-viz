import os
import joblib
import warnings
import numpy as np
import pandas as pd
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def _get_and_hadmid(path_model):
    all_files = os.listdir(path_model)
    all_files_emb = [file for file in all_files if file.split('_')[2] == "embeddings"]
    all_files_hadmid = [file for file in all_files if file.split('_')[2] == "hadmids"]

    hadmids_batch = {}
    for file in all_files_hadmid:
        batch = file.split('_')[4].split('.')[0] 
        id = joblib.load(path_model+file)[0]
        hadmids_batch[batch] = id
    print(f'hamdids_batch dict: {hadmids_batch}') ; print(f'embedding files: {all_files_emb}')
    return hadmids_batch, all_files_emb


def _get_non_padded_emb(path_model, all_files_emb=None, hadmids_batch=None, random_number_files=None, specific_hadmids=None, model=None): #working
    '''
    specific_hadmids: this is a dictionary with (batch, hadmids) = (k,v)
    '''
    model_emb_to_keep = {}
    hadmids_batch, all_files_emb = _get_and_hadmid(path_model)
    from random import sample
    if not random_number_files is None:
        sample_files_emb = sample(all_files_emb, random_number_files)
        all_files_emb = sample_files_emb
    if specific_hadmids is not None:
        emb_common_id = [emb for emb in all_files_emb if emb.split('_')[4].split('.')[0] in list(specific_hadmids.keys())]
        all_files_emb = emb_common_id
    for emb in all_files_emb:
        model_emb = joblib.load(path_model + emb) 
        batch = emb.split('_')[4].split('.')[0] ; hadmid = hadmids_batch[batch]
        if model == 'TCN' or model == 'Transformer' or model == 'TCN_att':
            model_emb_to_keep[hadmid] = model_emb[list(model_emb.keys())[0]] 
        else:
            model_emb_to_keep[hadmid] = model_emb
    return model_emb_to_keep

def _transform_non_padded_emb(path_model, random_number_files=None, specific_hadmids=None, model_emb_to_keep=None, model=None): #working
    model_emb_to_keep_df = {} ; 
    model_emb_to_keep = _get_non_padded_emb(path_model, random_number_files, specific_hadmids, model=model)
    for key in model_emb_to_keep.keys():
        model_emb_to_keep_df[key] = pd.DataFrame([key], columns=['hadmid']).merge(pd.DataFrame(model_emb_to_keep[key].cpu().numpy()), left_index=True, right_index=True, how='outer').ffill()
        model_emb_to_keep_df[key].hadmid = model_emb_to_keep_df[key].hadmid.astype(int)
        model_emb_to_keep_df[key].index.name = 'Time'
        model_emb_to_keep_df[key] = model_emb_to_keep_df[key].reset_index()
        model_emb_to_keep_df[key].Time = model_emb_to_keep_df[key].Time+1
        model_emb_to_keep_df[key] = model_emb_to_keep_df[key].set_index(['hadmid']).reset_index()
    return model_emb_to_keep_df



def plot_2d_decomp(decomp, figsize, LoS, title, save_path, plot_name, cdict, color_group, manifold_reducer=None, legend_name=None, adm_test_data=None, time=None): 
    plt.figure(figsize=figsize, facecolor = 'white')
    LoS = LoS.set_index('hadmid').merge(decomp, left_index=True, right_index=True, how='inner')
    df = LoS.merge(adm_test_data.set_index('hadmid'), left_index=True, right_index=True, how='inner')
    #df.to_csv(save_path + 'emb_df_'+str(time)+".csv", index=True) - Can use this to save emb dataframes
    gps = list(np.unique(LoS[color_group]))
    for g in np.unique(LoS[color_group]):
        ix = np.where(LoS[color_group] == g)
        
        if color_group == 'bins':
            if g == gps[0] or g == gps[1] or g == gps[2]:
                plt.scatter(LoS[f"{manifold_reducer}_1"].iloc[ix], LoS[f"{manifold_reducer}_2"].iloc[ix], c=cdict[g], label=g, marker="*", s=500)
            else:
                plt.scatter(LoS[f"{manifold_reducer}_1"].iloc[ix], LoS[f"{manifold_reducer}_2"].iloc[ix], c=cdict[g], label=g, marker="o", s=500)
        
        elif color_group == 'los_ihm':
            if g == "SA" or g == "SD":
                plt.scatter(LoS[f"{manifold_reducer}_1"].iloc[ix], LoS[f"{manifold_reducer}_2"].iloc[ix], c=cdict[g], label=g, marker="+", s=500)
            elif g == "MA" or g == "MD":
                plt.scatter(LoS[f"{manifold_reducer}_1"].iloc[ix], LoS[f"{manifold_reducer}_2"].iloc[ix], c=cdict[g], label=g, marker="d", s=500)
            else:#this is for LA & LD
                plt.scatter(LoS[f"{manifold_reducer}_1"].iloc[ix], LoS[f"{manifold_reducer}_2"].iloc[ix], c=cdict[g], label=g, marker="o", s=500)
                
        else:
            plt.scatter(LoS[f"{manifold_reducer}_1"].iloc[ix], LoS[f"{manifold_reducer}_2"].iloc[ix], c=cdict[g], label=g, s=500)
        plt.legend(loc = 'center left', title_fontsize=50, title = legend_name, fontsize=70, bbox_to_anchor=(1, 0.5)) 
        plt.xlabel('Dimension 1', fontsize=70) ; plt.ylabel('Dimension 2', fontsize=70)
        plt.xticks(fontsize=70) ; plt.yticks(fontsize=70)
        plt.title(title, fontsize=35)
        plt.show()
        plt.savefig(save_path + plot_name + ".png", bbox_inches='tight')    
    return 

def plot_tsne_decomp(emb_df, T, figsize, LoS, model_name, t, save_path, color_group, adm_test_data, scale_emb=None, manifold_reducer=None, legend_name=None):
    '''
    adm_test_data: admission data filtered on test patients
    '''
    if manifold_reducer =='tsne':
        reducer =  TSNE(n_components=3)
    elif manifold_reducer == 'umap':
        reducer = umap.UMAP(n_components=2)    
    elif manifold_reducer == 'pca':
        reducer = PCA(n_components=2)
    elif manifold_reducer == 'kpca':
        reducer = KernelPCA(n_components=2)
    elif manifold_reducer == 'tsvd':
        reducer = TruncatedSVD(n_components=2)
        
    if color_group == 'bins': #bins here stand for LoS grouped in 10 bins/classes
        bins = pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, 500)])
        labels = ['0-1day', '1-2days', '2-3days', '3-4days', '4-5days', '5-6days', '6-7days', '7-8days', '8-14days', 'over2weeks']
        LoS['bins'] = pd.cut(LoS.lengthofstay/24, bins=bins, labels=labels, precision=0)
        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'black', 'yellow', 'darkred', 'purple', 'grey']
        cdict = dict(zip(LoS.bins.unique(), colors)) 
    elif color_group == 'ihm':
        LoS = adm_test_data[['hadmid', 'ihm']].merge(LoS, on=['hadmid'], how='inner') ; print(f"shape of LoS with after merging with adm data:{LoS.shape}")
        colors = ['blue', 'red']
        cdict = dict(zip(LoS.ihm.unique(), colors))
    elif color_group == 'los_ihm':
        LoS = adm_test_data[['hadmid', 'ihm']].merge(LoS, on=['hadmid'], how='inner') ; print(f"shape of LoS with after merging with adm data:{LoS.shape}")
        LoS = create_los_ihm(LoS) ; 
        colors = ['blue', 'red', 'red', 'blue', 'red', 'blue']
        #colors = ['blue', 'magenta', 'green', 'olive', 'red', 'cyan'] #order here is SA(Short Alive), SD(Short Dead), LA(Long Alive), LD(Long Dead)
        cdict = dict(zip(LoS.los_ihm.unique(), colors))
    elif color_group == 'destination':
        LoS = adm_test_data[['hadmid', 'destination']].merge(LoS, on=['hadmid'], how='inner') ; print(f"shape of LoS with after merging with adm data:{LoS.shape}")
        LoS.destination.fillna("no_destination", inplace=True)
        colors = ['blue', 'orange', 'green', 'red', 'purple','brown','pink','gray','teal','darkblue',
                'grey', 'salmon', 'darkred', 'coral', 'lavender', 'black', 'forestgreen',
                'turquoise', 'chocolate', 'violet', 'indigo', 'saddlebrown', 'gold', 'olive', 'cyan', 'crimson',
                'magenta', 'lime', 'navy', 'deepskyblue', 'maroon', 'darkgoldenrod', 'aquamarine', 'rosybrown']
        cdict = dict(zip(LoS.destination.unique(), colors))
    elif color_group == 'Specialty':
        LoS = adm_test_data[['hadmid', 'specialty']].merge(LoS, on=['hadmid'], how='inner') ; print(f"shape of LoS with after merging with adm data:{LoS.shape}")
        LoS = group_specialty(adm_data = LoS)
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
        cdict = dict(zip(LoS.Specialty.unique(), colors))
    elif color_group == 'specialty':
        LoS = adm_test_data[['hadmid', 'specialty']].merge(LoS, on=['hadmid'], how='inner') ; print(f"shape of LoS with after merging with adm data:{LoS.shape}")
        LoS.specialty.fillna("no_specialty", inplace=True)
        colors = ['blue', 'orange', 'green', 'red', 'purple','brown','pink','gray','teal','darkblue',
               'grey', 'salmon', 'darkred', 'coral', 'lavender', 'black',
               'turquoise', 'forestgreen', 'indigo', 'violet', 'chocolate', 'gold', 'olive', 'cyan', 'crimson','magenta','lime']
        cdict = dict(zip(LoS.specialty.unique(), colors))
    warnings.filterwarnings("ignore")
    for time in (np.arange(1, T+t, t)):
        time_df = pd.DataFrame() ; #print(f"number of unique keys: {len(emb_df.keys())}") 
        for key in list(emb_df.keys()): 
            time_df = time_df.append(emb_df[key][(emb_df[key].hadmid == key) & (emb_df[key].Time == time)])
            df_to_project = time_df.drop(columns = ['Time']).set_index('hadmid')

        if not scale_emb:
            time_df_decomp = reducer.fit_transform(df_to_project)
        else:
            time_df_decomp = reducer.fit_transform(MinMaxScaler().fit_transform(df_to_project))
        time_df_decomp = pd.DataFrame(time_df_decomp, columns = [f"{manifold_reducer}_1", f"{manifold_reducer}_2", f"{manifold_reducer}_3"], index=df_to_project.index)
        print(f"{manifold_reducer} projection:{time_df_decomp} at time: {time}") #and indices: {time_df_decomp.index}") 
        plot_2d_decomp(decomp=time_df_decomp, figsize=figsize, LoS=LoS, title = f"{manifold_reducer} of {model_name} model at time {time}",
                                              save_path=save_path+"/", plot_name = model_name+"_"+str(time), cdict=cdict,
                                              color_group=color_group, manifold_reducer=manifold_reducer, legend_name=legend_name, time=time, adm_test_data=adm_test_data)       


def create_los_ihm(los):
    los['los_days'] = los.lengthofstay/24
    los['los_ihm'] = los.apply(lambda x:  "SD"  if (x['los_days']<=3 and x['mortality']==1)
                                            else ( "SA"  if (x['los_days']<=3 and x['mortality']==0)
                                            else ("MD" if (x['los_days'] > 3 and x['los_days'] <= 7 and x['mortality']==1)
                                            else ("MA" if (x['los_days'] > 3 and x['los_days'] <= 7 and x['mortality']==0)
                                            else ("LA" if (x['los_days']>7 and x['mortality']==0)
                                            else ( "LD" if (x['los_days']>7 and x['mortality']==1)
                                            else "empty"))))),axis=1)
    return los

def group_specialty(adm_data):
    adm_data['Specialty'] = adm_data['specialty']
    surgery = ['Cardiochirurgie', 'Neurochirurgie', 'Vaatchirurgie', 'Traumatologie',
               'Heelkunde Gastro-enterologie', 'Plastische chirurgie', 'Heelkunde Oncologie',
               'Urologie', 'Orthopedie', 'Heelkunde Longen/Oncologie']
    internal_med = ['Nefrologie', 'Cardiologie', 'Inwendig', 'Intensive Care Volwassenen',
                    'Neurologie', 'Longziekte', 'Hematologie', 'Maag-,Darm-,Leverziekten',
                     'Oncologie Inwendig']
    maternity = ['Gynaecologie', 'Obstetrie', 'Verloskunde']
    ophtamologie = ['Oogheelkunde', 'Keel, Neus & Oorarts']
    dentistry = ['Mondheelkunde']
    others = ['ders']
    adm_data['Specialty'] = adm_data['Specialty'].replace(surgery, 'Surgery')
    adm_data['Specialty'] = adm_data['Specialty'].replace(internal_med, 'Intern')
    adm_data['Specialty'] = adm_data['Specialty'].replace(maternity, 'Maternity')
    adm_data['Specialty'] = adm_data['Specialty'].replace(ophtamologie, 'Opthamology')
    adm_data['Specialty'] = adm_data['Specialty'].replace(dentistry, 'Dentistry')
    adm_data['Specialty'] = adm_data['Specialty'].replace(others, 'Others')
    adm_data.Specialty.fillna('Others', inplace=True)
    return adm_data
