#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################
import sys
import os
import joblib
import json
import warnings

import numpy as np
import seaborn as sns
import pandas as pd


from scipy.signal import resample_poly, resample, butter, filtfilt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence

from keras.layers import Input, Add, Dense, Activation, Flatten,  Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, BatchNormalization, Concatenate
from keras import ops

import keras.backend as K
K.set_image_data_format('channels_last')


from helper_code import *

#from model_ResNetCBAM_tf import resnet_CBAM
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from datetime import datetime

from joblib import Parallel, delayed




# define a common seed for reproducibility
global SEED
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################


#...............................................................................
# Train your model.
#...............................................................................
# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.
def train_model(data_folder, model_folder, verbose):
    # ...............................................................................
    # Fixed parameters
    # ...............................................................................  
    # (0) submission or development
    flag_development = False # True
    flag_debug = False # True
    
    # (3) Data splitting
    # (3.1) test set
    test_size = 0 # 0.2
    
    # (3.2) Number of stratified Kfold splits
    n_splits = 2

    # (3.3) Number of subsets for the dominant class
    num_subsets = 5

    # (4) Data generators
    signal_length = 3000 # 5000
    batch_size_train = 256 # 128
    batch_size_val = 256 # len(X_val) 

    # (5) Model definition
    number_residual_blocks = 8
    num_channels = 12
    learning_rate = 0.001
    reduce_lr_epochs = 20
    
    # (6) Training parameters (per subset)
    num_epochs = 150 

    # --------------------------------------------------------------------------
    # 0. Get the job id
    # --------------------------------------------------------------------------
    # get the job id to create an unique saving folder
    if flag_development:
        if flag_debug:
            job_id = 'job_debug'
        else:
            job_id = get_job_id()
        # update model_folder with the job id
        model_folder = os.path.join(model_folder, job_id)
    
    else:
        model_folder = model_folder
        test_size = 0

    # --------------------------------------------------------------------------
    # 1. Find the data files.
    # --------------------------------------------------------------------------
    if verbose:
        print('Finding the Challenge data...')
        print('Data folder: ', data_folder)
        print('Model folder: ', model_folder)

    # Dynamically get all subfolders in the data folder.
    subfolders = [f.name for f in os.scandir(data_folder) if f.is_dir()]

    # Filter out subfolders containing PTB_XL, ptb_xl, PTB-XL, or ptb-xl
    subfolders = [f for f in subfolders if not any(keyword in f for keyword in ['PTB_XL', 'ptb_xl', 'PTB-XL', 'ptb-xl'])]
    
    records_list = []
    
    #records_list = find_records(data_folder)
    #if not subfolders:
    records = find_records(data_folder)
    records_list.extend([f"{data_folder}/{record}" for record in records])
    #records_list = records
    #else:
    #    for subfolder in subfolders:
    #        path_signals = os.path.join(data_folder, subfolder)
    #        records = find_records(path_signals)

            # not the most efficient way to do it, but it takes 8.7 seconds to get all records
            #records_list.extend([f"{data_folder}/{subfolder}/{record}" for record in records])
            
    num_records = len(records_list)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')
    
    if verbose:
        print('Total records: ', len(records_list))
        print('Example file signal: ', records_list[1])
        #print(load_signals(records_list[1]))

    # --------------------------------------------------------------------------
    # 2. Extract (the features and) labels from the data.
    # --------------------------------------------------------------------------
    if verbose:
        print('Extracting features and labels from the data...')

    # get labels 
    labels = []
    #for file in records_list:
    #    label = load_label(os.path.join(data_folder, file + '.hea'))
    #    labels.append(label)

    labels = Parallel(n_jobs=-1)(
    delayed(load_label)(os.path.join( f + '.hea')) for f in records_list)

    if verbose:
        print('Total labels: ', len(labels))
        print('Example label: ', labels[1])

    #features = np.zeros((num_records, 6), dtype=np.float64)
    #labels = np.zeros(num_records, dtype=bool)

    # --------------------------------------------------------------------------
    # 3. Split data into training, test and validation sets
    # --------------------------------------------------------------------------
    # ...........................................................................
    # 3.1. Suppose/create an independent test set (excluded from signals loaded above)
    # ...........................................................................
    # Create a test set using the provided function
    if verbose:
        print('Creating test set...')

    test_set, test_labels, records_list, labels = create_test_set(records_list, labels, test_size=test_size)

    # Save the test set to a file
    if test_set!= []:
        save_test_set(data_folder, model_folder, test_set, test_labels)
        if verbose:
            print(f'Test set saved at: {model_folder}')

    
    # ...........................................................................
    # 3.2. Split data into training and validation sets.
    # ...........................................................................
    if verbose:
        print('Splitting data into training and validation sets...')
    
    # Prepare stratified k-fold splitting
    if verbose:
        print('Performing 5-fold stratified splitting...')
    
    # Extract additional features for balancing (age and sex) in a single loop
    stratification_features = []
    for file in records_list:
        header = load_header(os.path.join(file + '.hea'))
        age = get_age(header)
        sex = get_sex(header)
        sex_encoded = 0 if sex == 'Female' else 1 if sex == 'Male' else 2
        stratification_features.append([age, sex_encoded])
    
    # Convert to numpy arrays for stratification
    stratification_features = np.array(stratification_features)
    labels_array = np.array(labels)
    
    # Perform stratified k-fold splitting
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    nkfold = 0
    for train_index, val_index in mskf.split(records_list, np.column_stack((labels_array, stratification_features))):
        X_train, X_val = np.array(records_list)[train_index], np.array(records_list)[val_index]
        y_train, y_val = np.array(labels)[train_index], np.array(labels)[val_index]

        # Save train-validation splitting and visualize label distribution
        if verbose:
            print('Saving train-validation splitting and visualizing label distribution...')

        if flag_development:
            # save information regarding the splitting
            save_data_splitting(model_folder, stratification_features, 
                                train_index, val_index, 
                                X_train, X_val, y_train, y_val, 
                                kfold=nkfold)
            
        nkfold += 1
        break  # Use the first fold for training and validation
    
    # ......................................................................
    # 3.3. Data balancing
    # ......................................................................
    # Possible strategies:
    # - Oversampling: Duplicate samples from the minority class.
    # - Undersampling: Remove samples from the majority class.
    # - SMOTE: Generate synthetic samples for the minority class.
    # - Class weights: Assign higher weights to the minority class during training.
    # - Data augmentation: Apply transformations to the training data to increase diversity.
    # - Ensemble methods: Combine multiple models trained on different subsets of data.

    # 3.3.1. Create data subsets for the dominant class
    num_subsets = num_subsets
    # dominant class is the one with the most samples
    dominant_class = np.argmax(np.bincount(y_train.flatten()))
    dominant_class_subsets = create_ensemble_subsets(y_train, stratification_features,
                                                      num_subsets, verbose=True)
    
    if verbose:
        print(f"Number of subgroups created for dominant class: {len(dominant_class_subsets)}")
        for i, subgroup in enumerate(dominant_class_subsets):
            print(f"Subgroup {i + 1}: {len(dominant_class_subsets[i])} samples")


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # TRAIN A MODEL FOR EACH SUBSET
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    for n_subs in range(num_subsets):
        # ......................................................................
        # 3.3. Data balancing (continue)
        # ......................................................................   
        # Select training samples for the current subset using numpy indexing for efficiency
        subset_indices = dominant_class_subsets[n_subs]
        X_train_subset = np.take(X_train, subset_indices)
        y_train_subset = np.take(y_train, subset_indices)

        # merge dominant class samples with all the non-dominant class samples
        X_train_subset = np.concatenate((X_train_subset, X_train[y_train != dominant_class]))
        y_train_subset = np.concatenate((y_train_subset, y_train[y_train != dominant_class]))

        # 3.3.2. compute class weights
        class_weights = calculating_class_weights(y_train_subset)

        if verbose:
            # Print the number of samples of each class and their corresponding weights
            #class_counts = np.sum(y_train_subset, axis=0)
            for i in range(0, len(np.unique(y_train_subset))):
                class_counts = np.sum(y_train_subset == i)
                print(f"Class {i}: samples: {class_counts}, Weight: {class_weights[i]:.4f}")
        # --------------------------------------------------------------------------
        # 4. Define generators
        # --------------------------------------------------------------------------
        if verbose:
            print('Creating data generators...')
        
        # Create data generators
        signal_length = signal_length # 5000
        batch_size_train = batch_size_train # 128
        batch_size_val = batch_size_val # len(X_val) 
        

        train_generator = DataGenerator(X_train_subset, y_train_subset, signal_length, batch_size=batch_size_train)
        val_generator = DataGenerator(X_val, y_val, signal_length, batch_size=batch_size_val)

        if verbose:
            print('Train generator size: ', len(train_generator))
            print('Validation generator size: ', len(val_generator))

            # example generator
            #print('Example batch from train generator: ', X_train_subset[0])
            #print(train_generator[0])
        # --------------------------------------------------------------------------
        # 5. Define model
        # --------------------------------------------------------------------------
        if verbose:
            print('Building model...')

        # Define the model.
        number_residual_blocks = number_residual_blocks
        num_channels = num_channels
        learning_rate = learning_rate

        model_architecture = resnet_CBAM(N=number_residual_blocks, ch=num_channels, win_len=signal_length, classes=1)
        model = DLmodel(model=model_architecture, learning_rate=learning_rate, loss='binary_crossentropy', 
                        metrics=['accuracy', 'precision', 'recall'], reduce_lr_epochs=reduce_lr_epochs)
        
        # compile model
        model.compile()

        # --------------------------------------------------------------------------
        # 6. train model
        # --------------------------------------------------------------------------
        if verbose:
            print('Training model...')

        # Compute the number of steps per validation epoch
        validation_steps = int(np.ceil(len(X_val) / batch_size_val)) 

        # Train the model.
        num_epochs = num_epochs
        history_train = model.fit(train_generator, val_generator, batch_size=batch_size_train,
                                   epochs=num_epochs, validation_steps=validation_steps, 
                                   class_weights=class_weights)
        
    
        # --------------------------------------------------------------------------
        # 7. save partial models
        # --------------------------------------------------------------------------
        # Create a folder for the model if it does not already exist.
        os.makedirs(model_folder, exist_ok=True)

        # Save the model. --FLAG EDIT
        # Perhaps better to save all .keras models into a single dictionary to prevent loading errors.
        model_name = f'model_sbs_{n_subs}'
        #save_model_sbs(model_folder, model, model_name)
        model.save(model_folder, model_name)

        # Save training history as a JSON file.
        if verbose:
            print('Saving training history ...')
        

        history_plot_path = save_training_history(model_folder, history_train)
        if verbose:
            print('Training history saved at: ', history_plot_path)


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # COMBINE PARTIAL MODELS INTO AN ENSEMBLE MODEL
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # Combine the partial models into an ensemble model.
    if verbose:
        print('Combining partial models into an ensemble model using majority voting...')


    # Load all partial models
    partial_models = []
    for n_subs in range(num_subsets):
        model_path = os.path.join(model_folder, f'model_sbs_{n_subs}.keras')
        #partial_model = joblib.load(model_path)['model']
        partial_models.append(model_path)

    # Create the ensemble model
    ensemble_model = EnsembleModel(models=[], model_sbs_paths=partial_models)
    #ensemble_model = EnsembleModel(partial_models)
    ensemble_model.load() # not actually needed here, as we dont need to load models again, but used as check
    
    # --------------------------------------------------------------------------
    # 8. save final ensemble model
    # --------------------------------------------------------------------------
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    #save_model(model_folder, ensemble_model)
    ensemble_model.save(model_folder)
    
    if verbose:
        print('Ensemble model saved at:', model_folder)

    # Save a detailed diagram of the model structure.
    if verbose:
        print('Saving model structure diagram...')
    
    #model_structure_path = save_model_diagram(model_folder, model.model)
    #if verbose:
    #    print('Model structure saved at: ', model_structure_path)
    
  
    if verbose:
        print('Done.')
        print()

#...............................................................................
# Load your trained models.
#...............................................................................
# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.keras')
    
    # Load ensemble model
    metadata_path = os.path.join(model_folder, "ensemble_info.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            model_sbs_paths = json.load(f)
        ensemble_model = EnsembleModel(models=[], model_sbs_paths=model_sbs_paths)
        ensemble_model.load()
        model = ensemble_model
    else:
        # Fallback to single model loading
        model = keras.models.load_model(model_filename)
    return model

#...............................................................................
# Run your trained model.
#...............................................................................
# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Load the model.
    #model = model['model']
    try:
        # Load the ECG signal.
        # Load the header and signal data
        data = load_signals(record)

        # prepare the data
        signal_length = 3000 # 5000
        ecg_signal = data_preparation(data, signal_length)

        # Ensure the signal is in float32 format
        ecg_signal = ecg_signal.astype(np.float32)
        
        # add dimension for shape compatibility
        ecg_signal = np.expand_dims(ecg_signal, axis=0)

        # Get the model outputs.
        try:
            probability_output = model.predict_proba(ecg_signal) # TO BE IMPLEMENTED
            binary_output = model.predict(ecg_signal)
        except Exception as e:
            print(f"Error during prediction: {e}")
            binary_output = np.nan
            probability_output = np.nan
    
    except Exception as e:
        print(f"Error loading signals for {record}: {e}")
        binary_output = np.nan
        probability_output = np.nan
   
    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################


#....................................................................
# Data handling functions
#....................................................................
def extract_source(header: str) -> str:
    for line in header.splitlines():
        if line.startswith("# Source:"):
            return line[9:].strip()
    return ""

# extrat the database of each signal based of the abs file name
def _get_database_id(ecg_ids_abs):
    ecg_ids = []
    database_ids = []
    flag_from_path = False

    for ecg_id in ecg_ids_abs:
        if flag_from_path:
                if "SamiTrop" in ecg_id:
                    database_ids.append("SamiTrop")
                elif "CODE15" in ecg_id:
                    database_ids.append("CODE15")
                elif "PTB_XL" in ecg_id:
                    database_ids.append("PTB_XL")
                else:
                    database_ids.append("Unknown")
        else:
            # extract header from the file name
            try:
                header_temp = load_header(os.path.join(ecg_id + '.hea'))
                database_ids.append(extract_source(header_temp))
            except FileNotFoundError:
                # If the header file is not found, extract the database from the path
                database_ids.append("UNKNOWN")

    return ecg_ids, database_ids

# generate test set
def create_test_set(ecg_ids_abs, labels, test_size=0.2):
    # get database and ecg_ids from absolute ecg_ids paths
    _, database_id = _get_database_id(ecg_ids_abs)
    
    database_id = np.array(database_id)
    labels = np.array(labels)
    ecg_ids_abs = np.array(ecg_ids_abs)
    
    # exclude any signal in the database: PTB-XL
    # if the database is PTB-XL, exclude it from the test set
    #if 'PTB-XL' in database_id:
    #    ptb_xl_indices = np.where(database_id == 'PTB-XL')[0]
    #    database_id = np.delete(database_id, ptb_xl_indices)
    #    labels = np.delete(labels, ptb_xl_indices)
    #    ecg_ids_abs = np.delete(ecg_ids_abs, ptb_xl_indices)

    if test_size == 0:
        return [], [], ecg_ids_abs, labels
    # Stratify by labels for each database independently
    unique_databases = np.unique(database_id)
    test_indices = []

    for db in unique_databases:
        db_indices = np.where(database_id == db)[0]
        db_labels = labels[db_indices]

        # Perform stratified split for the current database
        _, db_test_indices = train_test_split(
            db_indices,
            test_size=test_size,
            random_state=SEED,
            stratify=db_labels
        )
        test_indices.extend(db_test_indices)

    # Create the test set
    test_indices = np.array(test_indices)
    test_set = ecg_ids_abs[test_indices]
    test_labels = labels[test_indices]

    # Create the remaining training set by excluding the test set
    train_indices = np.setdiff1d(np.arange(len(ecg_ids_abs)), test_indices)
    train_set = ecg_ids_abs[train_indices]
    train_labels = labels[train_indices]
    return test_set, test_labels, train_set, train_labels

# save data test
def save_test_set(data_folder, model_folder, test_set, labels):
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the test set to a file
    test_set_path = os.path.join(model_folder, 'test_set.json')
    with open(test_set_path, 'w') as f:
        json.dump(test_set.tolist(), f) 

    # Define the save path for the plot
    save_path = os.path.join(model_folder, 'test_distribution.png')

    # get database and ecg_ids from absolute ecg_ids paths
    _, database_ids = _get_database_id(test_set)

    # Plot and save the label distribution
    plot_database_distribution(database_ids, labels, 'Label Distribution by Database', save_path)

    # Plot test set demographics
    # Extract additional features for balancing (age and sex) in a single loop
    stratification_features = []
    
    for file in test_set:
        header = load_header(os.path.join(data_folder, file + '.hea'))
        age = get_age(header)
        sex = get_sex(header)
        sex_encoded = 0 if sex == 'Female' else 1 if sex == 'Male' else 2
        stratification_features.append([age, sex_encoded])
    
    # Convert to numpy arrays for stratification
    stratification_features = np.array(stratification_features)
    labels_array = np.array(labels)
    data_temp = pd.DataFrame({
        'Age': stratification_features[:, 0],
        'Sex': stratification_features[:, 1],
        'Label': labels_array
        })

    # Save demographic analysis plots
    demographic_analysis_path = os.path.join(model_folder, 'test_analysis.png')
    plot_temp = plot_demographic_analysis(data_temp)
    plot_temp.savefig(demographic_analysis_path)
    plot_temp.close()

    return save_path

# Extract your features.
def extract_features(record):
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)

    one_hot_encoding_sex = np.zeros(3, dtype=bool)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1
    else:
        one_hot_encoding_sex[2] = 1

    signal, fields = load_signals(record)

    # TO-DO: Update to compute per-lead features. Check lead order and update and use functions for reordering leads as needed.

    num_finite_samples = np.size(np.isfinite(signal))
    if num_finite_samples > 0:
        signal_mean = np.nanmean(signal)
    else:
        signal_mean = 0.0
    if num_finite_samples > 1:
        signal_std = np.nanstd(signal)
    else:
        signal_std = 0.0

    features = np.concatenate(([age], one_hot_encoding_sex, [signal_mean, signal_std]))

    return np.asarray(features, dtype=np.float32)

# data generator
class DataGenerator(Sequence):
    def __init__(self, ecg_filenames, labels, signal_length, batch_size=32, class_weights=None, shuffle=True): 
        super().__init__()
        self.ecg_filenames = ecg_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.signal_length = signal_length
        self.class_weights = class_weights
        self.shuffle = shuffle
        self.ecg_filter = self._ecg_filter(400)  # Assuming a default sampling frequency of 400 Hz
        # Create generators
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.ecg_filenames) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ecg_filenames))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # Get batch indexes
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_filenames = [self.ecg_filenames[k] for k in batch_indexes]
        batch_labels = [self.labels[k] for k in batch_indexes]

        # Generate data
        batch_signals = np.zeros((len(batch_filenames), self.signal_length, 12))
        batch_targets = np.zeros((len(batch_filenames), 1))
        for i, (fname, label) in enumerate(zip(batch_filenames, batch_labels)):
            try:
                data = load_signals(fname)
                signal = self._data_preparation(data)
            except Exception as e:
                print(f"Error loading signals for {fname}: {e}")
                signal = np.zeros((self.signal_length, 12))

            batch_signals[i] = signal
            batch_targets[i] = self._label_check(label)

        return batch_signals.astype(np.float32), batch_targets.astype(np.float32)

    def _label_check(self, label):
        try:
            # Check if label is directly in the set
            return 1 if label in {1, 1.0, '1'} else 0
        except TypeError:
            # Handle multi-dimensional or unhashable inputs
            return 0

    def _ecg_filter(self, fs):
        # Design a Butterworth bandpass filter
        lowcut = 1.0  # Lower cutoff frequency (Hz)
        highcut = 47.0  # Upper cutoff frequency (Hz)
        order = 3  # Filter order

        # Create the filter coefficients
        b, a = butter(order, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
        return b, a

    def _signal_preprocessing(self, ecg_signals, fs):
        """
        Signal preprocessing step
        input: ecg_signals: numpy array of shape (n_samples, n_leads)
               fs: int, sampling frequency
        output: ecg_signals: numpy array of shape (n_samples, n_leads)
        """
        # a. Resample
        if fs != 400:
            if fs == 1000:
                #  polyphase filtering for downsampling
                ecg_signals = resample_poly(ecg_signals, 2, 5, axis=0)
            else:
                #  FFT-based resampling for other sampling rates
                target_samples = int(ecg_signals.shape[0] * (400 / fs))
                ecg_signals = resample(ecg_signals, target_samples, axis=0, window='hamming')

        # b. filter
        # Apply the filter using zero-phase filtering
        #ecg_signals = filtfilt(self.ecg_filter[0], self.ecg_filter[1], ecg_signals, axis=0)

        # c. Normalize the signals: z-score
        #ecg_signals = (ecg_signals - np.mean(ecg_signals, axis=0)) / np.std(ecg_signals, axis=0)
        mean = np.mean(ecg_signals, axis=0)
        std = np.std(ecg_signals, axis=0)
        std[std == 0] = 1.0  # Evita división por cero
        ecg_signals = (ecg_signals - mean) / std
        
        return ecg_signals
    
    def _signals_armonization(self, ecg_signals):
        """Ensure signals have 12 leads and 5000 samples."""
        fixed_signal_length = self.signal_length
        n_rows, n_cols = ecg_signals.shape
        
        # Transpose if needed
        if n_cols != 12 and n_rows == 12:
            ecg_signals = ecg_signals.T

        # Ensure 12 leads
        if n_cols < 12:
            additional_cols = np.zeros((n_rows, 12 - n_cols))
            ecg_signals = np.hstack((ecg_signals, additional_cols))
        elif n_cols > 12:
            ecg_signals = ecg_signals[:, :12]

        # Ensure fixed number of samples
        if n_rows == 0:
            ecg_signals = np.zeros((fixed_signal_length, 12))
        elif n_rows < fixed_signal_length:
            additional_rows = np.zeros((fixed_signal_length - n_rows, 12))
            ecg_signals = np.vstack((ecg_signals, additional_rows))
        elif n_rows > fixed_signal_length:
            ecg_signals = ecg_signals[:fixed_signal_length, :]

        return ecg_signals
    
    def _data_preparation(self, data):
        """Preprocess and harmonize ECG signals."""
        ecg_signals = data[0]
        try:
            ecg_signals = data[0]
            try:
                ecg_signals = self._signal_preprocessing(data[0], data[1]['fs'])
            except Exception as e:
                print(f"Error in preprocessing: {e}")
                ecg_signals = data[0]

            try:
                ecg_signals = self._signals_armonization(ecg_signals)
            except Exception as e:
                print(f"Error in harmonization: {e}")
                ecg_signals = np.zeros((self.signal_length, 12))
        
        except Exception as e:
            ecg_signals = np.zeros((self.signal_length, 12))
        return ecg_signals
        
# data preprocessing

def _ecg_filter(self, fs):
    # Design a Butterworth bandpass filter
    lowcut = 1.0  # Lower cutoff frequency (Hz)
    highcut = 47.0  # Upper cutoff frequency (Hz)
    order = 3  # Filter order

    # Create the filter coefficients
    b, a = butter(order, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
    return b, a

def _signal_preprocessing(ecg_signals, fs):
    """
    Signal preprocessing step
    input: ecg_signals: numpy array of shape (n_samples, n_leads)
            fs: int, sampling frequency
    output: ecg_signals: numpy array of shape (n_samples, n_leads)
    """
    # a. Resample
    if fs != 400:
        if fs == 1000:
            #  polyphase filtering for downsampling
            ecg_signals = resample_poly(ecg_signals, 2, 5, axis=0)
        else:
            #  FFT-based resampling for other sampling rates
            target_samples = int(ecg_signals.shape[0] * (400 / fs))
            ecg_signals = resample(ecg_signals, target_samples, axis=0, window='hamming')

    # b. filter
    # Apply the filter using zero-phase filtering
    #ecg_signals = filtfilt(self.ecg_filter[0], self.ecg_filter[1], ecg_signals, axis=0)

    # c. Normalize the signals: z-score
    #ecg_signals = (ecg_signals - np.mean(ecg_signals, axis=0)) / np.std(ecg_signals, axis=0)
    mean = np.mean(ecg_signals, axis=0)
    std = np.std(ecg_signals, axis=0)
    std[std == 0] = 1.0  # Evita división por cero
    ecg_signals = (ecg_signals - mean) / std
    
    return ecg_signals

def _signals_armonization(ecg_signals, signal_length):
    """Ensure signals have 12 leads and 5000 samples."""
    fixed_signal_length = signal_length
    n_rows, n_cols = ecg_signals.shape
    
    # Transpose if needed
    if n_cols != 12 and n_rows == 12:
        ecg_signals = ecg_signals.T

    # Ensure 12 leads
    if n_cols < 12:
        additional_cols = np.zeros((n_rows, 12 - n_cols))
        ecg_signals = np.hstack((ecg_signals, additional_cols))
    elif n_cols > 12:
        ecg_signals = ecg_signals[:, :12]

    # Ensure fixed number of samples
    if n_rows == 0:
        ecg_signals = np.zeros((fixed_signal_length, 12))
    elif n_rows < fixed_signal_length:
        additional_rows = np.zeros((fixed_signal_length - n_rows, 12))
        ecg_signals = np.vstack((ecg_signals, additional_rows))
    elif n_rows > fixed_signal_length:
        ecg_signals = ecg_signals[:fixed_signal_length, :]

    return ecg_signals

def data_preparation(data, signal_length):
    """Preprocess and harmonize ECG signals."""
    ecg_signals = data[0]
    try:
        ecg_signals = data[0]
        try:
            ecg_signals = _signal_preprocessing(data[0], data[1]['fs'])
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            ecg_signals = data[0]

        try:
            ecg_signals = _signals_armonization(ecg_signals, signal_length)
        except Exception as e:
            print(f"Error in harmonization: {e}")
            ecg_signals = np.zeros((signal_length, 12))
    
    except Exception as e:
        ecg_signals = np.zeros((signal_length, 12))
    return ecg_signals


# compute class weights
def calculating_class_weights(y_true):
    # Compute class weights for binary classification
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_true.flatten())
    return {0: class_weights[0], 1: class_weights[1]}

# split data into training and validation sets
def split_data_train_val(ecg_filenames, labels, validation_size=0.2):
    """
    Function to split data into training and validation sets.
    To be implemented: stratify according to the labels but taking into consideration: age, sex, and other features (?).
    for example we could use the from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    """
    X_train, X_test, y_train, y_test = train_test_split(ecg_filenames, labels, test_size=validation_size, random_state=SEED, stratify=labels)
    return X_train, X_test, y_train, y_test

# save data splitting 
def save_data_splitting(model_folder, stratification_features, train_index, val_index, X_train, X_val, y_train, y_val, kfold=0):
    """
    Function to save data splitting into training and validation sets.
    """
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)  

    # Save the train-validation split
    split_path = os.path.join(model_folder, 'train_val_split.npz')
    np.savez(split_path, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)

    # Plot and save distributions for training set
    #train_dist_path = os.path.join(model_folder, 'train_label_distribution.png')
    #plot_label_distribution(y_train, stratification_features[train_index], 'Training Set', train_dist_path)
 
    # Plot and save distributions for validation set
    #val_dist_path = os.path.join(model_folder, 'val_label_distribution.png')
    #plot_label_distribution(y_val, stratification_features[val_index], 'Validation Set', val_dist_path)

    # Prepare data for demographic analysis using stratification_features -- FLAG: memory intensive
    # Convert stratification features to DataFrame
    data_temp = pd.DataFrame({
        'Age': stratification_features[val_index, 0],
        'Sex': stratification_features[val_index, 1],
        'Label': y_val
    })

    # Save demographic analysis plots
    demographic_analysis_path = os.path.join(model_folder, 'val_' + str(kfold) + 'Fold_analysis.png')
    plot_temp = plot_demographic_analysis(data_temp)
    plot_temp.savefig(demographic_analysis_path)
    plot_temp.close()

    data_temp = pd.DataFrame({
        'Age': stratification_features[train_index, 0],
        'Sex': stratification_features[train_index, 1],
        'Label': y_train
    })

    # Save demographic analysis plots
    demographic_analysis_path = os.path.join(model_folder, 'train_' + str(kfold) + 'Fold_analysis.png')
    plot_temp = plot_demographic_analysis(data_temp)
    plot_temp.savefig(demographic_analysis_path)
    plot_temp.close()

# emsemble subset creation
def create_ensemble_subsets(y_train, stratification_features, num_subsets, verbose=True):
    # Since we only have one binary class, calculate the count of positive and negative samples
    class_counts = [np.sum(y_train == 0), np.sum(y_train == 1)]
    dominant_class = np.argmax(class_counts)
    least_present_class = np.argmin(class_counts)

    if num_subsets is None:
        num_subsets = class_counts[dominant_class] // class_counts[least_present_class]
    
    if verbose:
        print(f"Dominant class: {dominant_class}, Count: {class_counts[dominant_class]}")
        print(f"Least present class: {least_present_class}, Count: {class_counts[least_present_class]}")
        print(f"Num subsets: {num_subsets}")

    # Divide the training samples of the dominant class into R subgroups, stratifying by age and sex
    dominant_class_indices = np.where(y_train == dominant_class)[0]
    np.random.shuffle(dominant_class_indices)

    # Extract age and sex for dominant class samples
    dominant_class_features = stratification_features[dominant_class_indices]
    dominant_class_subsets = create_dominant_class_subgroups(dominant_class_indices, dominant_class_features, num_subsets)

    return dominant_class_subsets

# create dominat class subgroups
def create_dominant_class_subgroups(dominant_class_indices, dominant_class_features, num_subsets):
    # Create subgroups using stratification
    subgroups = []
    subgroup_size = len(dominant_class_indices) // num_subsets

    # Create subgroups while maintaining the ratio of age
    for i in range(num_subsets):
        subgroup_indices = []

        # Stratify by age
        age_bins = pd.qcut(
            dominant_class_features[:, 0],
            q=min(4, len(dominant_class_indices)),  # Use up to 4 bins or fewer if not enough samples
            duplicates='drop',
            labels=False
        )

        for age_bin in np.unique(age_bins):
            bin_indices = dominant_class_indices[age_bins == age_bin]
            # Distribute samples evenly across subgroups
            samples_to_add = min(len(bin_indices), subgroup_size - len(subgroup_indices))
            subgroup_indices.extend(bin_indices[:samples_to_add])

        # Convert to numpy array and add to subgroups
        subgroup_indices = np.array(subgroup_indices)
        subgroups.append(subgroup_indices)

        dominant_class_features = dominant_class_features[np.isin(dominant_class_indices, subgroup_indices, invert=True)]
        dominant_class_indices = np.setdiff1d(dominant_class_indices, subgroup_indices, assume_unique=True)

        # Ensure subgroup size is met
        if len(subgroup_indices) < subgroup_size:
            remaining_indices = dominant_class_indices[:subgroup_size - len(subgroup_indices)]
            subgroup_indices = np.concatenate([subgroup_indices, remaining_indices])
            dominant_class_indices = dominant_class_indices[subgroup_size - len(subgroup_indices):]
        dominant_class_indices = np.setdiff1d(dominant_class_indices, subgroup_indices, assume_unique=True)

    return subgroups

         
#....................................................................
# Model functions
#....................................................................
# change it so that each of the models that we use are defined as classes
# (perhaps inside the tools/models) folder. Each will count with:
# - model_architecture: the model architectureç
# - model_compile: the model compilation
# - model_fitting: the model fitting
# - model_prediction: prediction of new labels

class DLmodel:
    def __init__(self, model, learning_rate=0.001, loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall'], reduce_lr_epochs=20):
        self.model = model
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.reduce_lr_epochs = reduce_lr_epochs
    
    def architecture(self):
        pass
    
    def compile(self):
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss=self.loss, metrics=self.metrics)
    
    def fit(self, data_generator_train, data_generator_val, batch_size=32, epochs=10, validation_steps=1, class_weights=None):
        # Add L2 regularization

        # Reduce learning rate by a factor of 0.1 after every 20th epoch
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: self.learning_rate * (0.1 ** (epoch // self.reduce_lr_epochs))
        )

        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.0001,
            patience=10,
            verbose=0,
            mode="auto", 
            baseline=None,
            restore_best_weights=False,
        )
        
        history_train = self.model.fit(
            data_generator_train,
            validation_data=data_generator_val,
            validation_steps=len(data_generator_val),
            class_weight=class_weights, 
            #shuffle=True,
            steps_per_epoch=len(data_generator_train),
            #batch_size=batch_size, # already handled by the generator
            epochs=epochs,
            callbacks=[callback, lr_scheduler]#, lr_scheduler]
        ).history
        return history_train
    
    def evaluate(self, data_generator_test, batch_size=32):
        results = self.model.evaluate(
            data_generator_test,
            steps=(len(data_generator_test.ecg_filenames)/batch_size),
            batch_size=batch_size,
            ) 
        return results

    def save(self, model_folder, model_name='model'):
        # Save the model architecture and weights
        model_path = os.path.join(model_folder, model_name + '.keras')
        self.model.save(model_path)
        return model_path
        
# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.keras')
    joblib.dump(d, filename, protocol=0)

# save trained model: partial models
def save_model_sbs(model_folder, model, model_name):
    # Save the model architecture and weights
    model.save(model_folder, model_name)

# Save model diagram    
def save_model_diagram(model_folder, model):
    model_structure_path = os.path.join(model_folder, 'model_structure.png')
    tf.keras.utils.plot_model(
        model.model,
        to_file=model_structure_path,
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=300
    )
    return model_structure_path

# Save training history
def save_training_history(model_folder, history_train):
    history_path = os.path.join(model_folder, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_train, f)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(history_train['loss'], label="Training Loss")
    plt.plot(history_train['val_loss'], label="Validation Loss")
    plt.title('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history_train['accuracy'], label="Training Accuracy")
    plt.plot(history_train['val_accuracy'], label="Validation Accuracy")
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(history_train['precision'], label="Training Precision")
    plt.plot(history_train['val_precision'], label="Validation Precision")
    plt.title('Precision')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(history_train['recall'], label="Training Recall")
    plt.plot(history_train['val_recall'], label="Validation Recall")
    plt.title('Recall')
    plt.legend()

    history_plot_path = os.path.join(model_folder, 'training_history.png')
    plt.savefig(history_plot_path)
    plt.close()

    return history_plot_path

class EnsembleModel:
    def __init__(self, models, model_sbs_paths):
        self.models = models
        self.flag_fine_tuning = False  # Flag to indicate if fine-tuning is done
        self.model_sbs_paths = model_sbs_paths
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        # Majority voting: take the mode of predictions across models
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x.astype(int), minlength=2).argmax(),
                                             axis=0, arr=(predictions > 0.5))
        return majority_vote
    
    def predict_proba(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        # Average probabilities across models
        average_probabilities = np.mean(predictions, axis=0)
        return average_probabilities

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        model_paths = []

        # save again subset models into single file
        if self.flag_fine_tuning:
            for i, model in enumerate(self.models):
                path = os.path.join(save_dir, f"model_{i}.keras")  # Use `.keras` or `.h5`
                model.save(path)
                model_paths.append(path)
            self.model_sbs_paths = model_paths

        # Save metadata (list of model paths)
        metadata_path = os.path.join(save_dir, "ensemble_info.json")
        with open(metadata_path, "w") as f:
            json.dump(self.model_sbs_paths, f)
        
    def load(self):
        # Load each model from the saved paths
        loaded_models = []
        for path in self.model_sbs_paths:
            model = keras.models.load_model(path)
            loaded_models.append(model)
        self.models = loaded_models

# load model subsets
def load_model_sbs(model_folder, model_name):
    # Load the model architecture and weights
    model_path = os.path.join(model_folder, model_name + '.keras')
    model = keras.models.load_model(model_path)
    return model

#....................................................................
# Visualization functions
#....................................................................

# plot training history
def plot_train_history(history):
    """
    Plot training history.
    """
    #plt.figure(figsize = (12,4))
    plt.subplot(2,2,1)
    plt.plot(history['loss'], label="training loss")
    plt.plot(history['val_loss'], label="validation loss")
    plt.title('Lossfunction ResNet:')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history['accuracy'], label="training accuracy")
    plt.plot(history['val_accuracy'], label="validation accuracy")
    plt.title('Accuracy ResNet:')
    plt.legend()

    #plt.figure(figsize = (12,4))
    plt.subplot(2,2,3)
    plt.plot(history['precision'], label="training precision")
    plt.plot(history['val_precision'], label="validation precision")
    plt.title('Precision ResNet:')
    plt.legend()

    plt.subplot(2,2,4)
    plt.plot(history['recall'], label="training recall")
    plt.plot(history['val_recall'], label="validation recall")
    plt.title('Recall ResNet:')
    plt.legend()

    return plt.show()

# visualize deep learning model structure
def visualize_model(model):

    tf.keras.utils.plot_model(
        model,
        to_file="model.png",
        show_shapes=False,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
        layer_range=None,
)


  # Visualize label distribution for each k-fold per age and sex


def plot_label_distribution(labels, stratification_features, title, save_path):
    ages = stratification_features[:, 0]
    sexes = stratification_features[:, 1]

    plt.figure(figsize=(12, 6))

    # Plot age distribution
    plt.subplot(1, 2, 1)
    plt.hist(ages, bins=20, alpha=0.7, label='Age')
    plt.title(f'{title} - Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')

    # Plot sex distribution
    plt.subplot(1, 2, 2)
    plt.bar([0, 1, 2], np.bincount(sexes), alpha=0.7, label='Sex')
    plt.title(f'{title} - Sex Distribution')
    plt.xlabel('Sex')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_demographic_analysis(data_df):
    """Analyze demographic features like Age and Sex"""
    # Age distribution by Label
    plt.figure(figsize=(15, 7))
    
    plt.subplot(2, 2, 1)
    sns.histplot(data=data_df, x='Age', hue='Label', kde=True, bins=20)
    plt.title('Age Distribution by Label')
    plt.xlabel('Age')
    plt.ylabel('Count')
    
    plt.subplot(2, 2, 2)
    sns.boxplot(data=data_df, x='Label', y='Age')
    plt.title('Age Box Plot by Label')
    plt.xlabel('Label')
    plt.ylabel('Age')
    
    #plt.tight_layout()
    #plt.show()
    
    # Sex distribution by Label
    #plt.figure(figsize=(15, 5))
    
    plt.subplot(2, 2, 3)
    sex_label = pd.crosstab(data_df['Sex'], data_df['Label'])
    ax = sex_label.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Sex Distribution by Label')
    plt.xlabel('Sex')
    plt.ylabel('Count')

    # Display total number per bar at the top of each bar
    for container in ax.containers:
        for rect in container:
            height = rect.get_height()
            if height > 0:
                if rect.get_y() + height > 0.5:  # Upper class
                    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + height + 0.1, f'{int(height)}', ha='center', va='bottom', fontsize=4)  # Add offset
                else:  # Lower class
                    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + height / 2 - 0.1, f'{int(height)}', ha='center', va='center', fontsize=4)  # Add offset
    
    plt.subplot(2, 2, 4)
    # Normalized stacked bar to see proportions
    sex_label_norm = sex_label.div(sex_label.sum(axis=1), axis=0)
    sex_label_norm.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Normalized Sex Distribution by Label')
    plt.xlabel('Sex')
    plt.ylabel('Proportion')
    
    plt.tight_layout()
    return plt


def plot_database_distribution(database_ids, labels, title, save_path):
    # Count the occurrences of each database and label combination
    database_label_counts = pd.crosstab(database_ids, labels, normalize='index') * 100

    # Plot the percentage of each database
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    database_counts = pd.Series(database_ids).value_counts(normalize=True) * 100
    database_counts.plot(kind='bar', color='skyblue', alpha=0.8)
    plt.title(f'{title} - Database Percentage')
    plt.xlabel('Database')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)

    # Plot the label distribution within each database (percentage)
    plt.subplot(2, 2, 2)
    database_label_counts.plot(kind='bar', stacked=True, colormap='viridis', alpha=0.8, ax=plt.gca())
    plt.title(f'{title} - Label Distribution by Database (Percentage)')
    plt.xlabel('Database')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot the absolute count of each database
    plt.subplot(2, 2, 3)
    database_counts_abs = pd.Series(database_ids).value_counts()
    database_counts_abs.plot(kind='bar', color='skyblue', alpha=0.8)
    plt.title(f'{title} - Database Count')
    plt.xlabel('Database')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    # Plot the label distribution within each database (absolute numbers)
    plt.subplot(2, 2, 4)
    database_label_counts_abs = pd.crosstab(database_ids, labels)
    database_label_counts_abs.plot(kind='bar', stacked=True, colormap='viridis', alpha=0.8, ax=plt.gca())
    plt.title(f'{title} - Label Distribution by Database (Count)')
    plt.xlabel('Database')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
#....................................................................
# other tools
#....................................................................

# jod id definition
def get_job_id():
    job_id = datetime.now().strftime("job_%Y%m%d_%H%M%S")
    if job_id is None:
        job_id = 'local'
    return job_id


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::MODEL ARCHITECTURE::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


#hint: use tf.reduce_mean(in_block, axis=()) for avg pooling 
# and tf.reduce_max(in_block, axis=()) for max pooling

def CBAM_block(in_block, ch, ratio=16):

    """
    @Convolutional Block Attention Module
    """

    #_, length, channel = in_block.get_shape()  # (B, L, C)

    # channel attention
    #avg_pool = tf.reduce_mean(in_block, axis=(1), keepdims=True)   # (B, 1, C)
    #max_pool = tf.reduce_max(in_block, axis=(1), keepdims=True)  # (B, 1, C)
    max_pool = ops.max(in_block, axis=1, keepdims=True)
    avg_pool = ops.mean(in_block, axis=1, keepdims=True) 

    dense1 = Dense(ch//ratio, activation='relu')
    avg_reduced = dense1(avg_pool) # (B, 1, C // r)
    max_reduced = dense1(max_pool) # (B, 1, C // r)

    dense2 = Dense(ch)
    avg_attention = dense2(avg_reduced) # (B, 1, C)
    max_attention = dense2(max_reduced) # (B, 1, C)

    #x = tf.add(avg_attention, max_attention)   # (B, 1, C)
    #x = tf.nn.sigmoid(x)        # (B, 1, C)
    #x = tf.multiply(in_block, x)   # (B, L, C)

    # spatial attention
    #y_mean = tf.reduce_mean(x, axis=-1, keepdims=True)  # (B, L, 1)
    #y_max = tf.reduce_max(x, axis=-1, keepdims=True)  # (B, L, 1)
    #y = tf.concat([y_mean, y_max], axis=-1)     # (B, L, 2)
    #y = tf.keras.layers.Conv1D(1, 7, padding='same', activation=tf.nn.sigmoid)(y)    # (B, L, 1)
    #y = tf.multiply(x, y)  # (B, L, C)


    # using keras
    x = ops.add(avg_attention, max_attention)   # (B, 1, C)
    x = ops.nn.sigmoid(x)        # (B, 1, C)
    x = ops.multiply(in_block, x)   # (B, L, C)

    # spatial attention
    y_mean = ops.mean(x, axis=-1, keepdims=True)  # (B, L, 1)
    y_max = ops.max(x, axis=-1, keepdims=True)  # (B, L, 1)
    y = Concatenate(axis=-1)([y_mean, y_max])     # (B, L, 2)
    y = Conv1D(1, 7, padding='same', activation=ops.nn.sigmoid)(y)    # (B, L, 1)
    y = ops.multiply(x, y)  # (B, L, C)

    return y
# ResNet with Convolutional Block Attention Module definition 


def ResBs_CBAM_Conv(block_input, num_filters): 
   
    # 0. Filter Block input and BatchNormalization
    block_input_short = Conv1D(num_filters, kernel_size=7, strides=2,  padding = 'valid')(block_input) 
    block_input_short = BatchNormalization()(block_input_short)

    # 1. First Convolutional Layer
    conv1 = Conv1D(filters=num_filters, kernel_size=7, strides=2, padding= 'valid')(block_input)
    norm1 = BatchNormalization()(conv1)
    relu1 = Activation('relu')(norm1)  
    dropout = Dropout(0.2)(relu1)
    
    # 2. Second Convolutional Layer 
    conv2 = Conv1D(num_filters, kernel_size=7, padding= 'same')(dropout) #per avere concordanza
    norm2 = BatchNormalization()(conv2)

    # 3. CBAM block (fucntion defined above)
    CBAM = CBAM_block(norm2, ch=num_filters)

    # 4. Summing Layer (adding a residual connection)
    sum = Add()([block_input_short, CBAM])
    
    # 5. Activation Layer
    relu2 = Activation('relu')(sum)
    
    return relu2 

def ResBs_CBAM_Identity(block_input, num_filters): 

    # 1. First Convolutional Layer
    conv1 = Conv1D(filters=num_filters, kernel_size=7, padding= 'same')(block_input)
    norm1 = BatchNormalization()(conv1)
    relu1 = Activation('relu')(norm1)    
    dropout = Dropout(0.2)(relu1)
    
    # 2. Second Convolutional Layer 
    conv2 = Conv1D(num_filters, kernel_size=7, padding= 'same')(dropout) #per avere concordanza
    norm2 = BatchNormalization()(conv2)

    # 3. CBAM block (fucntion defined above)
    CBAM = CBAM_block(norm2, ch=num_filters)

    # 4. Summing Layer (adding a residual connection)
    sum = Add()([block_input, CBAM])
    # 5. Activation Layer
    relu2 = Activation('relu')(sum)
    
    return relu2 

# model integrating deep + wide 
def resnet_CBAM(N=8, ch=12, win_len=5000, classes=9): 
    # B. ECG window input of shape (batch_size,  WINDOW_LEN, CHANNELS
    ecg_input = Input(shape=(win_len, ch), name='ecg_signal') 

    ResNet = Conv1D(filters=64,kernel_size=15, padding = 'same')(ecg_input) 
    ResNet = BatchNormalization()(ResNet)
    ResNet = Activation('relu')(ResNet)
    ResNet = MaxPooling1D(pool_size=2, strides = 2)(ResNet)
    
    # B.5 ResBs (x8) blocks
    # The number of filters starts from 64 and doubles every two blocks
    
    # First two ResNet blocks are identity blocks 
    
    ResNet = ResBs_CBAM_Identity(ResNet, 64)
    ResNet = ResBs_CBAM_Identity(ResNet, 64)

    filters = 64
    M= int((N -2 )/2)
    for i in range(M): 
        filters = filters*2

        # define N-th ResBs block
        ResNet = ResBs_CBAM_Conv(ResNet, filters)
        ResNet = ResBs_CBAM_Identity(ResNet, filters)
    
    ResNet = GlobalMaxPooling1D(name='gmp_layer')(ResNet)
    ResNet = Flatten()(ResNet)

    ResNet = Dense(classes, activation='sigmoid', name='sigmoid_classifier')(ResNet)
   
    # Finally the model is composed by connecting inputs to outputs: 
    model = Model(inputs=ecg_input, outputs=ResNet)

    return model