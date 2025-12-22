import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Subject IDs in WESAD data
subject_ids = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10",
               "S11", "S13", "S14", "S15", "S16", "S17"]

# Data modality labels in WESAD data
data_labels = ["ECG", "EDA", "EMG", "Temp", "Resp"]

# File path to WESAD data
data_file_path = os.getenv("WESAD_PATH")

# File path for saving images
save_path_imgs = os.path.join(os.getenv("SAVE_PATH"), "Images")

# File path for saving data with only good labels
save_path_good_labels = os.path.join(os.getenv("SAVE_PATH"), "Good_labels")

# File path for saving pre-processed data
save_path_preproc = os.path.join(os.getenv("SAVE_PATH"), "Pre-processed_data")


def load_data(subject_id, file_path):
    """
    This function loads data into a numpy array with 6 rows, ECG, EDA, EMG,
    TEMP and RESP data, and the last row is the corresponding label

    Parameters:
        subject_id: id of the subject whose data is to be loaded
        file_path: file path to the loadable data
    Returns:
        a numpy array with 6 rows
    """
    path = os.path.join(file_path, subject_id, f'{subject_id}.pkl')
    try:
        with open(path, 'rb') as file:
            data = pickle.load(file, encoding="latin1")
            file.close()
    except FileNotFoundError:
        print(f"File not found: {path}")
    except pickle.UnpicklingError:
        print("Error: The file content is not a valid pickle format.")
    except EOFError:
        print("Error: The file is incomplete or corrupted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    data_stack = [np.squeeze(np.array(data['signal']['chest']['ECG'])),
                  np.squeeze(np.array(data['signal']['chest']['EDA'])),
                  np.squeeze(np.array(data['signal']['chest']['EMG'])),
                  np.squeeze(np.array(data['signal']['chest']['Temp'])),
                  np.squeeze(np.array(data['signal']['chest']['Resp'])),
                  np.squeeze(np.array(data['label']))
                  ]
    return np.array(data_stack)


def pre_process_slice_to_min(data, data_len, data_height, min_length, subject_id):
    """
    This function slices one subject's data from beginning into the minimum
    timeseries length of all the WESAD subjects

    Parameters:
        data: subject's timeseries data with corresponding labels
        data_len: length of one row
        data_height: number of rows
        min_length: minimum timeseries length of all the WESAD subjects
        subject_id: id of the subject whose data is to be sliced
    Returns:
        a sliced numpy array of shape (data_height, min_length)
    """
    print(f"Slicing {subject_id}'s data from the beginning")
    return_data = data[0:data_height, data_len - min_length:data_len]
    print(f"{subject_id}'s data shape: {return_data.shape}")
    return return_data


def pre_process_minmax(data, data_len, data_height):
    """
    This function normalizes the timeseries values between [-1,1]

    Parameters:
        data: timeseries data with corresponding labels
        data_len: length of one row
        data_height: number of rows
    Returns:
        data with timeseries values normalized between [-1,1]
    """
    data_np = np.array(data)
    mins = np.min(data_np, axis=1)
    maxes = np.max(data_np, axis=1)
    divs = maxes - mins
    print(f"Mins: {mins}")
    print(f"Maxs: {maxes}")
    print(f"Divs: {divs}")

    return_data = [[None for _ in range(data_len)] for _ in range(data_height)]
    for i in range(0, data_height - 1):
        for j in range(0, data_len):
            return_data[i][j] = 2 * (data[i][j] - mins[i]) / divs[i] - 1
    for j in range(0, data_len):
        return_data[data_height - 1][j] = data[data_height - 1][j]

    return np.array(return_data)


def remove_bad_labels(data, data_len, data_height):
    """
    This function removes bad labels and corresponding timeseries data points,
    and merges the rest together

    Parameters:
        data: timeseries data with corresponding labels
        data_len: length of one row
        data_height: number of rows
    Returns:
        data with only desired labels and corresponding timeseries points
    """
    # Delete data points corresponding to labels 0, 4, 5, 6, 7

    wanted_labels = [1, 2, 3]

    # Loop through the data and append only wanted labels and corresponding
    # data points
    data_sliced = [[] for _ in range(data_height)]
    for j in range(0, data_len):
        if data[data_height-1][j] in wanted_labels:
            for i in range(data_height):
                data_sliced[i].append(data[i][j])

    return np.array(data_sliced)


def change_labels(data, data_len, data_height):
    """
    This function changes labels to 0 and 1, 1=stress

    Parameters:
        data: timeseries data with only labels 1, 2, 3
        data_len: length of one row
        data_height: number of rows
    Returns:
        data with timeseries values normalized between [-1,1]
    """
    for j in range(0, data_len):
        if data[data_height - 1][j] == 2:
            data[data_height - 1][j] = 1
        else:
            data[data_height - 1][j] = 0
    return data


def draw_data(data, subject_id):
    """
    This function draws and saves images of cleaned data

    Parameters:
        data: timeseries data with corresponding labels
        subject_id: id of the subject whose data is to be visualized
    Returns:
        None
    """
    for i in range(0, len(data_labels)):
        plt.figure()
        plt.title(f"Cleaned data, {data_labels[i]}-data")
        plt.plot(np.arange(0, len(data[i])), np.array(data[i]))
        dir_img = os.path.join(save_path_imgs, subject_id)
        if not os.path.exists(dir_img):
            os.makedirs(dir_img)
        save_pth = os.path.join(dir_img, f"{data_labels[i]}_cleaned_data.pdf")
        # Delete old image if it exists
        if os.path.exists(save_pth):
            os.remove(save_pth)
        # Save the image
        plt.savefig(os.path.join(dir_img, f"{data_labels[i]}_cleaned_data.pdf"))
        plt.close()


def draw_data_with_bad(data, data_height, index_list, subject_id):
    """
    This function draws and saves images of uncleaned data

    Parameters:
        data: timeseries data with corresponding labels
        data_height: number of rows
        index_list: list of indexes where the label value changes
        subject_id: id of the subject whose data is to be visualized
    Returns:
        None
    """
    for i in range(0, len(data_labels)):
        fig, ax1 = plt.subplots()
        fig.set_figheight(10)
        fig.set_figwidth(50)
        ax1.plot(np.arange(0, len(data[i])), np.array(data[i]))
        ax1.set_xlabel("Time, one step 1/700 seconds")
        ax1.set_ylabel(f"{data_labels[i]}-data value")

        # Collect label changes into an array
        label_changes = []
        for j in index_list:
            plt.axvline(x=j, color='g')
            label_changes.append(f"{int(data[data_height-1][j-1])}-"
                                 f"{int(data[data_height-1][j+1])}")

        ax2 = ax1.twiny()
        ax2.set_xlabel("Label change")
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(index_list)
        ax2.set_xticklabels(label_changes)
        ax2.set_title(f"Data with bad data ranges, {data_labels[i]}-data",
                      fontweight='bold',
                      pad=20)

        dir_img = os.path.join(save_path_imgs, subject_id)
        if not os.path.exists(dir_img):
            os.makedirs(dir_img)
        save_pth = os.path.join(dir_img, f"{data_labels[i]}_data_with_bad.pdf")
        # Delete old image if it exists
        if os.path.exists(save_pth):
            os.remove(save_pth)
        # Save the image
        plt.savefig(save_pth)
        plt.close()


def find_index_list(data, data_len, data_height):
    """
    This function finds indexes where the label changes

    Parameters:
        data: timeseries data with corresponding labels
        data_len: length of one row
        data_height: number of rows
    Returns:
        List of indexes where the label changes
    """

    index_list = []

    j = 1

    while j < data_len:
        if data[data_height - 1][j-1] != data[data_height - 1][j]:
            index_list.append(j-1)
        j += 1

    return index_list


def pre_process_whole():
    """
    This function handles the whole pre-processing, removing unnecessary labels,
    calculating the minimum timeseries length of all subjects and slicing the
    data of all subjects into this minimum length, and normalizing the modality
    values between [-1, 1]. Images of the modality plots are saved before and
    after the pre-processing.

    Returns:
        None
    """
    length_list = []
    for sub in subject_ids:
        print(f"Processing {sub} data:")
        data = load_data(sub, data_file_path)
        data_height, data_len = data.shape
        print(f"Number of rows: {data_height}, number of columns: {data_len}")
        # Draw and save images of uncleaned data
        draw_data_with_bad(data, data_height, find_index_list(data, data_len,
                                                              data_height), sub)

        # Remove bad labels and save data lengths
        print(f"Removing unnecessary labels of {sub} data")
        data_cleaned = remove_bad_labels(data, data_len, data_height)
        data_height, data_len = data_cleaned.shape
        length_list.append(data_len)

        print(f"Number of rows: {data_height}, number of columns: {data_len}\n")

        # Save data with only good labels
        dir_save = os.path.join(save_path_good_labels, sub)
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        np.save(os.path.join(dir_save, f"{sub}_pre_processed_rmv_bad_labels.npy"), data_cleaned)

    # Calculate minimum length of all subjects
    min_len = None
    for l in length_list:
        if min_len is None:
            min_len = l
        elif l < min_len:
            min_len = l
    print(f"Minimum length of all subjects: {min_len}\n")

    # Rest of pre-processing, slicing data into same size and
    # performing minmax normalization
    for sub in subject_ids:
        print(f"Pre-processing {sub} data:")
        dir_save = os.path.join(save_path_good_labels, sub)
        path = os.path.join(dir_save, f"{sub}_pre_processed_rmv_bad_labels.npy")
        try:
            with open(path, 'rb') as file:
                data_cleaned = np.load(file)
                file.close()
        except FileNotFoundError:
            print(f"File not found: {path}")
        except pickle.UnpicklingError:
            print("Error: The file content is not a valid pickle format.")
        except EOFError:
            print("Error: The file is incomplete or corrupted.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        data_height, data_len = data_cleaned.shape
        data_cleaned = pre_process_slice_to_min(data_cleaned, data_len, data_height, min_len, sub)
        data_height, data_len = data_cleaned.shape
        data_cleaned = pre_process_minmax(data_cleaned, data_len, data_height)
        data_height, data_len = data_cleaned.shape
        print(f"Number of rows: {data_height}, number of columns: {data_len}")
        draw_data(data_cleaned, sub)

        plt.close('all')

        # Change labels
        data_cleaned = change_labels(data_cleaned, data_len, data_height)
        # Check if unwanted labels are still present
        wanted_labels = np.array([0, 1])
        still_bad = False
        for i in range(0, data_len):
            if data_cleaned[data_height - 1][i] not in wanted_labels:
                print(f"Label: {data_cleaned[data_height - 1][i]}, index : {i}")
                still_bad = True
                break
        if still_bad:
            print("Still bad data")
        else:
            print("Data labels changed successfully\n")

        # Save data
        dir_save = os.path.join(save_path_preproc, sub)
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        np.save(os.path.join(dir_save, f"{sub}_pre_processed_data.npy"), data_cleaned)

    print("Pre-processing is complete.")

