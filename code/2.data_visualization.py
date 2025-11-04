import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import os

#file paths
dst1_hdf5 = "/Users/whynot.son/Downloads/Dataset/chunk1.hdf5"
dst1_csv  = "/Users/whynot.son/Downloads/Dataset/chunk1.csv"
dst2_hdf5 = "/Users/whynot.son/Downloads/Dataset/chunk2.hdf5"
dst2_csv  = "/Users/whynot.son/Downloads/Dataset/chunk2.csv"

#read CSV
chunk_1 = pd.read_csv(dst1_csv)
chunk_2 = pd.read_csv(dst2_csv)
full_csv = pd.concat([chunk_1, chunk_2], ignore_index=True)

#Save path
SAVE_PATH = "/Users/whynot.son/Downloads/Test/results_hehe"
os.makedirs(SAVE_PATH, exist_ok=True)

#draw histogram
def histogram(num_cols):
    full_csv[num_cols].hist(bins=50, figsize=(20,15))
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, "histogram.png"))
    plt.show()

#draw correlation matrix
def corel_mtx(num_cols):
    corr_matrix = full_csv[num_cols].corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, "co_rel.png"))
    plt.show()

# Phân phối dữ liệu categorical
def data_distribution(top_n=20):
    cat_cols = ['network_code', 'receiver_type', 'p_status', 's_status',
                'source_magnitude_type', 'source_magnitude_author', 'trace_category']
    for col in cat_cols:
        if col in full_csv.columns:
            plt.figure(figsize=(14,7))

            # riêng network_code thì gom top N
            if col == "network_code":
                vc = full_csv[col].value_counts()
                top_vc = vc.head(top_n)
                others = vc.iloc[top_n:].sum()
                if others > 0:
                    top_vc.loc["Others"] = others
                top_vc.plot(kind="bar")
                plt.title(f"{col} (Top {top_n} + Others)")
            else:
                full_csv[col].value_counts().plot(kind='bar')
                plt.title(col)

            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_PATH, f"{col}.png"))
            plt.show()

#main
if __name__ == "__main__":
    num_cols = ['receiver_latitude', 'receiver_longitude', 'receiver_elevation_m', 'p_arrival_sample',
                'p_weight', 'p_travel_sec', 's_arrival_sample', 's_weight',
                'source_latitude', 'source_longitude',
                'source_magnitude', 'source_distance_deg',
                'source_distance_km', 'back_azimuth_deg',
                ]
    data_distribution(top_n=45)  #can change
    histogram(num_cols)
    corel_mtx(num_cols)

def waveform_spectrogram_plot(signal_path,signal_index,signal_list):
    dtfl = h5py.File(signal_path, 'r') #find the signal file
    dataset = dtfl.get('data/'+str(signal_list[signal_index])) #fetch one signal from the file
    #waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    data = np.array(dataset)

    #plot
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(9,7))
    ax1.plot(np.linspace(0,60,6000),data[:,2],color='k',linewidth=1) #plot waveform
    ymin, ymax = ax1.get_ylim()
    ax1.vlines(dataset.attrs['p_arrival_sample']/100,ymin,ymax,color='b',linewidth=1.5, label='P-arrival') # plot p-wave arrival time
    ax1.vlines(dataset.attrs['s_arrival_sample']/100, ymin, ymax, color='r', linewidth=1.5, label='S-arrival') # plot s-wave arrival time
    ax1.vlines(dataset.attrs['coda_end_sample']/100, ymin, ymax, color='cyan', linewidth=1.5, label='Coda end')
    ax1.set_xlim([0,60])
    ax1.legend(loc='lower right',fontsize=10)
    ax1.set_ylabel('Amplitude (counts)')
    ax1.set_xlabel('Time (s)')
    im = ax2.specgram(data[:,2],Fs=100,NFFT=256,cmap='jet',vmin=-10,vmax=25) # plot spectrogram
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    ax3.psd(data[:,2],256,100,color='cornflowerblue') # plot power spectral density
    ax3.set_xlim([0,50])
    plt.savefig('waveform_spectrogram_plot.png',dpi=500)
    plt.tight_layout()
    plt.show()

    print('The p-wave for this waveform was picked by: ' + dataset.attrs['p_status'])
    print('The s-wave for this waveform was picked by: ' + dataset.attrs['s_status'])

dst2_csv_df = pd.read_csv(dst2_csv)
waveform_spectrogram_plot(dst2_hdf5, 12000, dst2_csv_df['trace_name'].to_list())