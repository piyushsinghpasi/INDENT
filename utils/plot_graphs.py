import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from pathlib import Path
# import matplotlib.font_manager
# matplotlib.font_manager._rebuild()
# from matplotlib import rc
# rc('text', usetex=True)


import seaborn as sns 

def plot_chunks(all_chunk_scores, all_seg_ids, GT_seg_id, GT_Q_text, top_5_text,
    GT_rank, save_path = ""):
    '''
    all_chunk_scores : list of chunks score parallel to all_seg_ids
    '''
    
    new_seg_ids = []
    max_colors = 35
    colors = sns.color_palette('pastel', max_colors)[:max_colors]
    GT_color = 'black'
    # GT_color = sns.color_palette('bright', 1)[0]
    # print(GT_color)
    # print(colors[:4])
    seg_color = []
    widths = []
    for i, chunk in enumerate(all_chunk_scores):
        new_seg_ids += [all_seg_ids[i]]*(chunk.shape[0])

        # curr_color = 'r' if all_seg_ids[i] == GT_seg_id else 'b'
        curr_color = GT_color if all_seg_ids[i] == GT_seg_id else colors[i%max_colors]
        w = 0.5 if all_seg_ids[i] == GT_seg_id else 0.5

        widths += [w]*(chunk.shape[0])
        seg_color += [curr_color]*(chunk.shape[0])
        
    
    chunk_scores = np.concatenate(all_chunk_scores).ravel()
    # print(chunk_scores[:10], new_seg_ids[:10])
    fig, ax = plt.subplots(1, 1, figsize = (16,8))
    # fig, (ax, a1) = plt.subplots(1, 2, width_ratios=[3, 1], figsize = (16,8))

    x_axis = range(len(new_seg_ids))
    ax.bar(x_axis, chunk_scores, color=seg_color, width=widths)
    

    # ax.set_xticks(x_axis)
    # ax.set_xticklabels(new_seg_ids)
    ax.set_xticks([])
    ax.set_xticks([], minor=True)

    max_len = 10
    txt = "GT: {:.3f}: {}\n ".format(GT_rank, all_chunk_scores[:5])
    # for i, qt in enumerate(top_5_text):
    #     txt += " \n{}-Seg ID: {:.3f}\n ".format(top_5_scores[i][0], top_5_scores[i][1])
    #     for q in qt:
    #         q_1 = q.strip().replace("\\n", " ").replace("$", "#")
                # len_q_1 = len(q_1)
                # start, i = 0, 1
                # while (len_q_1 > max_len):
                #     tmp = q_1[start:i*max_len]
                #     start = max_len
                #     len_q_1 -= max_len
                #     i += 1
                #     txt += tmp[]
                # txt += q_1[start:]
            # txt += f"{q_1}"
            # txt += " \n\n "
        

    nirm = Path('./fonts/Nirmala.ttf')
    hindi_font = FontProperties(fname = nirm)
    GT_legend = mpatches.Patch(color=GT_color, label='Ground Truth Segment')
    ax.legend(handles=[GT_legend])

    # a1.set_axis_off()
    # a1.text(-0.4, -0.6,
    #     txt, 
    #     fontproperties=hindi_font,
    #     # wrap = True,
    # )
    ax.set_ylabel('Similarity Scores')
    ax.set_xlabel('Chunks')
    ax.set_title('Similarity Scores for all Chunks and Question')
    # fig.tight_layout()
    # fig.canvas.mpl_connect('draw_event', on_draw)
    plt.savefig(save_path)
    print("saved at", save_path)
    plt.close()
