import plotly
import torch
import numpy as np
import pandas as pd
import plotly.graph_objs as go

import plotly.express as px

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from itertools import cycle

import os
import sys
sys.path.append(os.path.dirname("./"))
from dataloader_e2e_speech_batched import *
from nets.net_e2e import Feat_Merger
import random
random.seed(1)


def display_tsne_scatterplot_3D(seg_chunk_feat, user_input=[], words=None, label=None, color_map=None, perplexity = 0, learning_rate = 0, iteration = 0, topn=1, sample=10):
    """_summary_

    Args:
        seg_chunk_feat (dict): segment-ID : segment chunks
        user_input (_type_, optional): _description_. Defaults to None.
        words (_type_, optional): _description_. Defaults to None.
        label (_type_, optional): _description_. Defaults to None.
        color_map (_type_, optional): _description_. Defaults to None.
        perplexity (int, optional): _description_. Defaults to 0.
        learning_rate (int, optional): _description_. Defaults to 0.
        iteration (int, optional): _description_. Defaults to 0.
        topn (int, optional): _description_. Defaults to 5.
        sample (int, optional): _description_. Defaults to 10.
    """
    
    word_vectors = []
    user_input = []
    topN = []
    words = []
    for seg_id, feat in seg_chunk_feat.items():
        N, D = feat.size()
        words += [seg_id]*N
        
        user_input += [seg_id]
        topN += [N]
        word_vectors += [feat.detach().cpu().numpy()]
        
    word_vectors = np.concatenate(word_vectors)
    # print(word_vectors.shape)
    
    label_dict = dict([(y,x+1) for x,y in enumerate(set(words))])
    color_map = [label_dict[x] for x in words]

    three_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:3]
    # three_dim = TSNE(n_components = 3, random_state=0, perplexity = perplexity, learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:3]

    
    seg_id = pd.DataFrame(words, columns=['seg_id'])
    out_df = pd.DataFrame(three_dim,columns=['x', 'y', 'z'])
    
    out_df = pd.concat([out_df, seg_id], axis=1)
    print(out_df)
    unique_segids = len(list(set(words)))
    colors = px.colors.qualitative.Light24 + px.colors.qualitative.Dark24
    random.shuffle(colors)
    colors = colors[: unique_segids ]
        
    plot = px.scatter_3d(out_df, x = 'x', 
                     y = 'y',
                     z='z',
                     color='seg_id',
                     text = 'seg_id',
                    #  color_continuous_scale=px.colors.sequential.Rainbow,
                     color_discrete_sequence=colors,
    )
    plot.show()
    
    return
    # exit(0)

    # For 2D, change the three_dim variable into something like two_dim like the following:
    # two_dim = TSNE(n_components = 2, random_state=0, perplexity = perplexity, learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:2]

    color = 'blue'
    
    data = []

    count = 0
    colors = cycle(plotly.colors.sequential.Viridis)
    print("co", len(list(colors)))

    for i in range (len(user_input)):
                topn = topN[i]
                trace = go.Scatter3d(
                    x = three_dim[count:count+topn,0], 
                    y = three_dim[count:count+topn,1],  
                    z = three_dim[count:count+topn,2],
                    text = words[count:count+topn],
                    name = user_input[i],
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': next(colors),
                        # 'colorscale':'azure',
                    }
                )
                
                # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
                # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
            
                data.append(trace)
                count = count+topn

    trace_input = go.Scatter3d(
                    x = three_dim[count:,0], 
                    y = three_dim[count:,1],  
                    z = three_dim[count:,2],
                    text = words[count:],
                    name = 'input words',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 1,
                        'color': color_map,
                    },
                    )

    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
            
    # data.append(trace_input)
    
# Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 1000
        )


    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.show()
    
def tsneQ(word_vectors, user_input=[], words=None, label=None, color_map=None, perplexity = 0, learning_rate = 0, iteration = 0, topn=1, sample=10):
    """_summary_

    Args:
        seg_chunk_feat (dict): segment-ID : segment chunks
        user_input (_type_, optional): _description_. Defaults to None.
        words (_type_, optional): _description_. Defaults to None.
        label (_type_, optional): _description_. Defaults to None.
        color_map (_type_, optional): _description_. Defaults to None.
        perplexity (int, optional): _description_. Defaults to 0.
        learning_rate (int, optional): _description_. Defaults to 0.
        iteration (int, optional): _description_. Defaults to 0.
        topn (int, optional): _description_. Defaults to 5.
        sample (int, optional): _description_. Defaults to 10.
    """
    
    topN = [1]*len(user_input)
    words = user_input
    
    label_dict = dict([(y,x+1) for x,y in enumerate(set(words))])
    color_map = [label_dict[x] for x in words]

    # three_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:3]
    three_dim = TSNE(n_components = 3, random_state=0, perplexity = perplexity, learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:3]
# 

    # For 2D, change the three_dim variable into something like two_dim like the following:
    # two_dim = TSNE(n_components = 2, random_state=0, perplexity = perplexity, learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:2]

    color = 'blue'
    
    data = []

    count = 0
    for i in range (len(user_input)):
                topn = topN[i]
                trace = go.Scatter3d(
                    x = three_dim[count:count+topn,0], 
                    y = three_dim[count:count+topn,1],  
                    z = three_dim[count:count+topn,2],
                    text = words[count:count+topn],
                    name = user_input[i],
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 0.8,
                        'color': 3
                    }
                )
                
                # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
                # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
            
                data.append(trace)
                count = count+topn

    trace_input = go.Scatter3d(
                    x = three_dim[count:,0], 
                    y = three_dim[count:,1],  
                    z = three_dim[count:,2],
                    text = words[count:],
                    name = 'input words',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 10,
                        'opacity': 1,
                        'color': color_map,
                    },
                    )

    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
            
    # data.append(trace_input)
    
# Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 1000
        )


    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.show()
    
if __name__ == "__main__":
    
    model = Feat_Merger()
    
    model.load_state_dict(torch.load("./models/" + "e2e-speech-Gaussian-Loss-1-Self-1-Cross-1e-3-Sim-B4-E40-N4-lr3e-4-beta-1e-1" + ".pt"))
    model.eval()
    
    test_segs_dir = "../care_india_data/audio_hindi_val_segments/"
    test_files = set()
    for f in os.listdir(test_segs_dir) :
        if os.path.isdir(os.path.join(test_segs_dir,f)):
            filename = f.split("___")[0]
            test_files.add(filename)

    tot_R_1, tot_R_5, tot_R_10, tot_R_avg = 0., 0., 0., 0.
    for file_name in test_files:
        question_dataset = Question_dataset(
            file_name=file_name, 
            # test_file='./../care_india/audio_features_dict_traindata_5ques.json',
        )
        question_loader = DataLoader(
            question_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=1, 
            pin_memory = True
        )
        
    word_vector = []
    user_input = [] 
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(question_loader):

            segment_scores = dict()
            Question_feat, GT_seg_id = sample['Question'], (sample['segment_id'])[0]

            Q_feat = model(None, Question_feat, None, None, None, None, None)
            Q_feat = Q_feat.squeeze().unsqueeze(0)
            word_vector += [Q_feat.numpy()]
            
        word_vectors = np.concatenate(word_vector)
    tsneQ(word_vectors, user_input, None, None, None, 5, 500, 10000)