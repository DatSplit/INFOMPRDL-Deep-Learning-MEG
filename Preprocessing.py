import os
import h5py
import numpy as np
import mne
import os
import re


TASK_LABELS = {
    'rest': 0,
    'task_working_memory': 1,
    'task_motor': 2,
    'task_story_math': 3
}
TASKS = ['Intra', 'Cross']


ROOT_DIRECTORIES = {
    'Intra': 'data/Intra',
    'Cross': 'data/Cross'
}



# Run this block if you want to preprocess all the files in the dataset.
def extract_dataset_name(file_path):
    # file path by '\' (Windows)
    components = file_path.split(os.sep)
    
    # Get the filename from the path
    filename = components[-1]
    
    # Use regular expression to find the specific patterns
    match = re.search(r'(rest|task_working_memory|task_motor|task_story_math)_\d+', filename)
    
    if match:
        dataset_name = match.group(0)
        return dataset_name
    
    return None  # Return None if the pattern is not found

def extract_task_name(file_path):
    
    match = re.search(r'(rest|task_working_memory|task_motor|task_story_math)_\d+', file_path)
    print(match.group(1))
    if match:
        return match.group(1)
    
    return None

# We should scale the data based on the avg and std of the entire data (train / test split) not just the file we are looking at
# So I calculated this for each folder before so we dont need to load all files to do it :) 
mean = {
  'data/Intra/test': 1.0700527814994439e-12,
  'data/Intra/train': 8.121469141775592e-13,
  'data/Cross/test1': 1.91983681104587e-13,
  'data/Cross/train': 7.475291134724408e-14,
  'data/Cross/test3': -9.668145789473494e-14,
  'data/Cross/test2': -9.567592835020362e-13
}
std_dev = {
  'data/Intra/test': 1.4186728247260929e-11,
  'data/Intra/train': 9.29474180235638e-12,
  'data/Cross/test1': 8.860733266114578e-12,
  'data/Cross/train': 9.697619214580789e-12,
  'data/Cross/test3': 1.4209202524121167e-11,
  'data/Cross/test2': 1.6583664443436448e-11
}

def preprocess_MEG_data(root_dir, task):
    output_dir = 'Preprocessed'
    output_dir_scale = 'Preprocessed_scale'
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.h5'):
                file_path = os.path.join(dirpath, filename)
                
                relative_path = os.path.relpath(file_path, root_dir)

                output_path_scale = os.path.join(output_dir_scale, task, relative_path)
                output_dirname_scale = os.path.dirname(output_path_scale)
                os.makedirs(output_dirname_scale, exist_ok=True)
                npy_output_path_scale = output_path_scale.replace('.h5', '.npz')

                with h5py.File(file_path, 'r') as f:
                    dataset_name = extract_dataset_name(fr'{file_path}')
                    print(dataset_name)
                    data = f.get(dataset_name)[:]
                
                    channel_names = ['Ch' + str(i + 1) for i in range(data.shape[0])]
                    channel_types = ['mag' for i in range(data.shape[0])]
                    sfreq = 2034
                    info = mne.create_info(ch_names=channel_names, ch_types=channel_types, sfreq=sfreq)
                    raw = mne.io.RawArray(data,info)
                    
                    freqs = (60, 120)
                    # Notch filter to remove power line noise at 60 Hz and 120 Hz.
                    raw_notch = raw.copy().notch_filter(freqs=freqs)

                    data = raw_notch.get_data()
                    data = np.array(data)
                    # Z-Score Normalization
                    X_standardized = (data.T - mean[dirpath.replace('\\', '/')]) / std_dev[dirpath.replace('\\', '/')]
                    # Resample for computational efficiency
                    # Low pass filter to reduce frequencies above 333 Hz.
                    X_standardized = X_standardized.T
                    channel_names = ['Ch' + str(i + 1) for i in range(X_standardized.shape[0])]
                    channel_types = ['mag' for _ in range(X_standardized.shape[0])]
                    info = mne.create_info(ch_names=channel_names, ch_types=channel_types, sfreq=sfreq)
                    standardized = mne.io.RawArray(X_standardized,info)
                    raw_notch_lowpass = standardized.copy().filter(l_freq=None, h_freq=333)
                    raw_downsampled = raw_notch_lowpass.copy().resample(sfreq=1000)
                    

                    data = raw_downsampled.get_data()
                    data = np.array(data).T
                    print(data.shape)

                    # Extract task name from file path
                    current_task = extract_task_name(file_path)
                    label = TASK_LABELS[current_task]
                    #Save preprocessed data as numpy array.
                    print(npy_output_path_scale)
                    np.savez(npy_output_path_scale, data=data, task=current_task, label=label)



def main():
    # Process files for each task
    print('Starting preprocessing')
    for task in TASKS:
        root_directory = ROOT_DIRECTORIES[task]
        preprocess_MEG_data(root_directory, task)

if __name__ == "__main__":
    main()