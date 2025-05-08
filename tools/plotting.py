import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.stats import mannwhitneyu

from tools.utilities import timeStructured
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

def plot_predictions(sound2sound, spike2sound, generator_test_snd2snd, generator_test_spk2snd, ex):
    # test spike to sound

    batch_spk_test, batch_snd_test = generator_test_spk2snd.__getitem__()

    predicted_sound = spike2sound.predict_on_batch(batch_spk_test)
    one_spike, one_sound, one_predicted_sound = batch_spk_test[0], batch_snd_test[0], predicted_sound[0]

    fig, axs = plt.subplots(3)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(one_spike)
    axs[0].set_title('spike')
    axs[1].plot(one_sound)
    axs[1].set_title('sound')
    axs[2].plot(one_predicted_sound)
    axs[2].set_title('predicted sound')

    fig_path = 'model/spk2snd.pdf'
    fig.savefig(fig_path, bbox_inches='tight')
    ex.add_artifact(fig_path)

    # test sound to sound
    batch_snd_input, batch_snd_output = generator_test_snd2snd.__getitem__()

    predicted_sound = sound2sound.predict_on_batch(batch_snd_input)
    one_sound_input, one_sound_output, one_predicted_sound = batch_snd_input[0], batch_snd_output[0], predicted_sound[0]

    fig, axs = plt.subplots(3)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(one_sound_input)
    axs[0].set_title('input sound')
    axs[1].plot(one_sound_output)
    axs[1].set_title('output sound')

    axs[2].plot(one_predicted_sound)
    axs[2].set_title('predicted sound')

    fig_path = 'model/snd2snd.pdf'
    fig.savefig(fig_path, bbox_inches='tight')
    ex.add_artifact(fig_path)


def plot_losses(n2n_lh, k2n_lh, n2n_lh_cpc, k2n_lh_cpc, ex):
    # plot training losses
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(n2n_lh, label='n2n_lh')
    ax.plot(k2n_lh, label='k2n_lh')
    ax.plot(n2n_lh_cpc, label='n2n_lh_cpc')
    ax.plot(k2n_lh_cpc, label='k2n_lh_cpc')
    ax.set_title('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend()

    random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
    plot_filename = 'model/{}_train_losses.pdf'.format(random_string)
    fig.savefig(plot_filename, bbox_inches='tight')
    ex.add_artifact(plot_filename)


def plot_final_bar_plot(list_terms, final_scores, plot_filename):
    # compare the performance of the different approaches
    y_pos = np.arange(len(list_terms))

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.bar(y_pos, final_scores, align='center', alpha=0.5)
    # ax.xticks(y_pos, list_terms)
    ax.set_xticklabels(list_terms, fontdict=None, minor=False)

    ax.set_ylabel('MSE')
    ax.set_title('training type')

    fig.savefig(plot_filename, bbox_inches='tight')


def plot_test(prediction, batch_input, batch_snd_out_test, exp_type, image_title, fig_path, batch_sample):
    if 'WithSpikes' in exp_type:
        one_sound = batch_input[0][batch_sample]
    elif 'noSpike' in exp_type:
        one_sound = batch_input[batch_sample]
    else:
        raise NotImplementedError

    one_sound_out, one_predicted_sound = batch_snd_out_test[batch_sample], prediction[batch_sample]

    fig, axs = plt.subplots(3)
    fig.suptitle(image_title)
    axs[0].plot(one_sound)
    axs[0].set_title('input sound')
    axs[1].plot(one_sound_out)
    axs[1].set_title('output sound')
    axs[2].plot(one_predicted_sound)
    axs[2].set_title('predicted sound')

    fig.savefig(fig_path, bbox_inches='tight')
    plt.close('all')


def one_plot_test(prediction, clean, noisy, image_title, fig_path):
    fig, axs = plt.subplots(3)
    fig.suptitle(image_title)
    axs[0].plot(noisy[0])
    axs[0].set_title('noisy sound')
    axs[1].plot(clean[0])
    axs[1].set_title('clean sound')
    axs[2].plot(prediction[0])
    axs[2].set_title('predicted sound')
    plt.subplots_adjust(wspace=0.9, hspace=0.9)
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close('all')


def save_wav(prediction, noisy_sound, clean_sound, exp_type, batch_sample, fs, images_dir, subject=None,
             generator_type=None):
    if not 'unattended' in generator_type:
        
        clean_sound_path = os.path.join(*[images_dir, 'clean_{}_b{}_s{}.wav'.format(exp_type, batch_sample, subject)])
        noisy_sound_path = os.path.join(*[images_dir, 'noisy_{}_b{}_s{}.wav'.format(exp_type, batch_sample, subject)])
        predicted_sound_path = os.path.join(
            *[images_dir, 'prediction_{}_b{}_s{}_g{}.wav'.format(exp_type, batch_sample, subject, generator_type)])
        
        m = np.max(np.abs(noisy_sound[0]))
        one_sound_in32 = (noisy_sound[0] / m).astype(np.float32)
        wavfile.write(noisy_sound_path, int(fs), one_sound_in32)
    
        m = np.max(np.abs(clean_sound[0]))
        one_sound_out32 = (clean_sound[0] / m).astype(np.float32)
        wavfile.write(clean_sound_path, int(fs), one_sound_out32)
    
        m = np.max(np.abs(prediction[0]))
        m = m if not m == 0 else 1
        one_predicted_sound32 = (prediction[0] / m).astype(np.float32)
        wavfile.write(predicted_sound_path, int(fs), one_predicted_sound32)

    else:
        clean_sound_path = os.path.join(*[images_dir, 'clean_{}_b{}_s{}_g{}.wav'.format(exp_type, batch_sample, subject,generator_type)])
        m = np.max(np.abs(clean_sound[0]))
        one_sound_out32 = (clean_sound[0] / m).astype(np.float32)
        wavfile.write(clean_sound_path, int(fs), one_sound_out32)




def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1] + 1, height[num1]
    rx, ry = center[num2] + 1, height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y + barh, y + barh, y]
    mid = ((lx + rx) / 2, y + barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
   # plt.rc('font',**{'family':'serif','serif':['Times']})
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    # ax.xaxis.xticks(rotation=90)
    ax.set_xticklabels(labels, size=12)# rotation=25)
    ax.set_xlim(0.25, len(labels) + 0.75)
    #ax.set_yticks(np.array([0,4,8,12]))#([4,6,8,10]))
    ax.tick_params('y',labelsize=18)
    #ax.rc('xtick', labelsize=20) 
    #ax.set_ylim(-10,20)
    #plt.grid()
    plt.rcParams["font.family"] = "serif"
    #plt.rcParams["font.serif"] = "Times New Roman"
    #plt.rcParams["font.size"]=6
    # ax.set_xlabel('Sample name')
    # plt.xticks(rotation=3)


def evaluations_to_violins(all_exps, images_folder, name_suffix='',all_subjects='yes',noise='yes'):
    print(all_exps.keys())
    last_exp = list(all_exps.keys())[-1]
    # metrics = [m for m in list(all_exps[last_exp].columns) if 'noisy' not in m]
    metrics = list(all_exps[last_exp].columns)

    desired_order_list = [key for key in all_exps.keys() if key!='Mixture']
    if all_subjects=='no' and noise=='yes':
        name_m=['Mixture']
        for m in desired_order_list:
            name_m.append(m)
        
        desired_order_list=name_m

    reordered_dict = {k: all_exps[k] for k in desired_order_list}
    all_exps=reordered_dict
    
    
    fig = plt.figure(figsize=(12,12))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    kj = 0
    
    for m in metrics:
        if kj < 4 and kj != 2:
            print(m)
            data = []

            for k, v in all_exps.items():
                # print(k, v)
                data.append(list(v[m]))

            # data.append(list(v[m + '_noisy']))
            colors = plt.cm.gist_earth(np.linspace(0.2, .8, len(data)))
            # colors = ['lightyellow', 'thistle', 'forestgreen', 'lightskyblue']
            if kj == 3:
                ax = fig.add_subplot(3, 1, kj)
            else:
                ax = fig.add_subplot(3, 1, kj + 1)
            # ax.set_title(m)
            plt.locator_params(nbins=3)
            ax.set_ylabel(m.upper(), fontsize=16)
            parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

            for pc, c in zip(parts['bodies'], colors):
                pc.set_facecolor(c)
                pc.set_edgecolor('black')
                pc.set_alpha(1)

            quartile1, medians, quartile3 = [], [], []
            for data_sample in data:
                quartile1_i, medians_i, quartile3_i = np.percentile(data_sample, [25, 50, 75], axis=0)
                quartile1.append(quartile1_i)
                medians.append(medians_i)
                quartile3.append(quartile3_i)

            whiskers = np.array([
                adjacent_values(sorted(sorted_array), q1, q3)
                for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
            whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]


            inds = np.arange(1, len(medians) + 1)
            ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
            ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
            ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

            # set style for the axes

            #for key in all_exps.keys():
                #if 'Subject' in key:
               #     print(key)
              #      key_old=key
             #       all_exps[key.replace('Subject','')] = all_exps.pop(key_old)
            all_exps_new={}
            if all_subjects=='yes':
                for key in all_exps.keys():
                    if 'Subject' in key:
                        print(key)
                        key_old=key
                        all_exps_new[key.replace('Subject','')] = all_exps[key_old]

                labels = all_exps_new.keys()
            else:
                labels = all_exps.keys()

            set_axis_style(ax, labels)
            upper_labels = [str(np.round(s, 2)) for s in medians]
            pos = np.arange(len(data)) + 1

            for tick, label in zip(range(len(data)), ax.get_xticklabels()):
                ax.text(pos[tick], 1.01, upper_labels[tick],
                        transform=ax.get_xaxis_transform(),
                        horizontalalignment='center', fontsize=13, rotation=45, fontweight='bold') #, rotation='vertical')  # ,
            #                    fontweight='bold')

            maxs_data = np.array([max(d) for d in data])
            v_space = 0
            p_all=[]
            for i, di in enumerate(data):
                for j, dj in enumerate(data):
                    if j > i:
                        v_space += .1
                        _, p = mannwhitneyu(di, dj)
                        #print(p)
                        p_all.append(p)
                        #barplot_annotate_brackets(i, j, p, range(len(data)), maxs_data + v_space)
            if 'PESQ' in m.upper():
                plt.yticks([1.0, 1.5, 2.0, 2.5])
            if 'STOI' in m.upper():
                plt.yticks([0.3, 0.5, 0.7, 0.9])
            if 'SI-SDR' in m.upper():
                plt.yticks([-5, 0, 5, 10, 15])
            plt.tick_params(axis='y', labelsize=16)
        kj = kj + 1
    
    plt.show()
    time_string = timeStructured()
    import json
    #path_p=os.path.join(images_folder, 'p_{}-metrics_subjects_{}_{}.json'.format(time_string, m, name_suffix))

    #with open(path_p, "w") as f:   
     #   json.dump(p_all, f,indent=2)

    fig.savefig(
        os.path.join(images_folder, 'testfig.pdf'),
        bbox_inches='tight'
    )

   

'''        
def evaluations_to_violins(all_exps, images_folder, name_suffix='',all_subjects='yes',noise='yes'):
    print(all_exps.keys())
    last_exp = list(all_exps.keys())[-1]
    # metrics = [m for m in list(all_exps[last_exp].columns) if 'noisy' not in m]
    metrics = list(all_exps[last_exp].columns)

    desired_order_list = [key for key in all_exps.keys() if key!='Mixture']
    if all_subjects=='no' and noise=='yes':
        name_m=['Mixture']
        for m in desired_order_list:
            name_m.append(m)
        
        desired_order_list=name_m

    reordered_dict = {k: all_exps[k] for k in desired_order_list}
    all_exps=reordered_dict

    for m in metrics:
        print(m)
        data = []

        for k, v in all_exps.items():
            # print(k, v)
            data.append(list(v[m]))

        # data.append(list(v[m + '_noisy']))
        colors = plt.cm.gist_earth(np.linspace(0.2, .8, len(data)))
        # colors = ['lightyellow', 'thistle', 'forestgreen', 'lightskyblue']

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4), sharey=True)
        # ax.set_title(m)
        plt.locator_params(nbins=3)
        ax.set_ylabel(m.upper(), fontsize=16)
        parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

        for pc, c in zip(parts['bodies'], colors):
            pc.set_facecolor(c)
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        quartile1, medians, quartile3 = [], [], []
        for data_sample in data:
            quartile1_i, medians_i, quartile3_i = np.percentile(data_sample, [25, 50, 75], axis=0)
            quartile1.append(quartile1_i)
            medians.append(medians_i)
            quartile3.append(quartile3_i)

        whiskers = np.array([
            adjacent_values(sorted(sorted_array), q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]


        inds = np.arange(1, len(medians) + 1)
        ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

        # set style for the axes
        
        #for key in all_exps.keys():
            #if 'Subject' in key:
           #     print(key)
          #      key_old=key
         #       all_exps[key.replace('Subject','')] = all_exps.pop(key_old)
        all_exps_new={}
        if all_subjects=='yes':
            for key in all_exps.keys():
                if 'Subject' in key:
                    print(key)
                    key_old=key
                    all_exps_new[key.replace('Subject','')] = all_exps[key_old]
            
            labels = all_exps_new.keys()
        else:
            labels = all_exps.keys()
            
        set_axis_style(ax, labels)
        upper_labels = [str(np.round(s, 2)) for s in medians]
        pos = np.arange(len(data)) + 1

        for tick, label in zip(range(len(data)), ax.get_xticklabels()):
            ax.text(pos[tick], 1.01, upper_labels[tick],
                    transform=ax.get_xaxis_transform(),
                    horizontalalignment='center', fontsize=10, rotation=45) #, rotation='vertical')  # ,
        #                    fontweight='bold')

        maxs_data = np.array([max(d) for d in data])
        v_space = 0
        p_all=[]
        for i, di in enumerate(data):
            for j, dj in enumerate(data):
                if j > i:
                    v_space += .1
                    _, p = mannwhitneyu(di, dj)
                    #print(p)
                    p_all.append(p)
                    #barplot_annotate_brackets(i, j, p, range(len(data)), maxs_data + v_space)

        plt.show()
        time_string = timeStructured()
        import json
        path_p=os.path.join(images_folder, 'p_{}-metrics_subjects_{}_{}.json'.format(time_string, m, name_suffix))
        
        with open(path_p, "w") as f:   
            json.dump(p_all, f,indent=2)
            
        fig.savefig(
            os.path.join(images_folder, '{}-metrics_subjects_{}_{}.png'.format(time_string, m, name_suffix)),
            bbox_inches='tight'
        )
        
        fig.savefig(
            os.path.join(images_folder, '{}-metrics_subjects_{}_{}.eps'.format(time_string, m, name_suffix)),
            bbox_inches='tight',format='eps'
        )'''

