# Project assignment: Environmental audio analysis
# Student: Sijan Pandey(H293831)
# Email:sijan.pandey@tuni.fi

import librosa
import matplotlib.pyplot as plt
import numpy as np
import csv
import sklearn.metrics.pairwise as cos_module

CSV = "H293831.csv"
LABELS = ['adults_talking', 'dog_barking', 'footsteps',
          'traffic_noise', 'birds_singing', 'siren',
          'music', 'children_voices', 'announcement_speech']
def load_file():
    files = []
    filelabels = []
    with open(CSV) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0] != "fileName":
                labels = row[1].split(',')
                # get labels for each audio file
                filelabels.append(labels)
                name = row[0]
                # some error from courseside: audio files were entered with .mp3 extension
                # which actually are in .wav format
                name = name.replace(".mp3", ".wav")
                # get audio file names
                files.append(name)
    return files, filelabels

# get the feature vector from MFCC
def features(audio):
    x, fs = librosa.load('audio_files/' + audio, sr=None)
    # about 20 ms frame, 50% overlap
    mfcc = librosa.feature.mfcc(x, fs, n_mfcc=40, hop_length=512, n_fft=1024)
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    feature = np.concatenate([mean, std])
    return feature

# calculate and plot cosine similarity matrix
def similarity(feature_list):
    similarity_matrix = cos_module.cosine_similarity(feature_list) # sklearn library
    plt.imshow(similarity_matrix, origin='lower')
    plt.colorbar()
    plt.xlabel('Audio sample numbers')
    plt.ylabel('Audio sample numbers')
    plt.title('Cosine similarity of the audio samples')
    plt.show()
    print('Average of similarity matrix:', np.mean(similarity_matrix))
    return similarity_matrix

# get the average of similarity for each class from similarity matrix
def class_similarity(similarity_matrix, filelabels):
    with open('class_similarity.csv', 'w', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow(['class_labels', 'class_similarity_mean'])
        # get all the files numbers tuple that has the given label
        for i in range(len(LABELS)):
            elements = []
            class_values = []
            for j in range(len(filelabels)):
                for k in range(len(filelabels)):
                    if LABELS[i] in filelabels[j] and LABELS[i] in filelabels[k]:
                        if j != k:
                            elements.append(sorted([j, k]))
            # remove duplicate tuple elements from the list
            elements = list(set(map(lambda l: tuple(sorted(l)), elements)))
            for m in range(0, len(elements)):
                class_values.append(similarity_matrix[elements[m][0]][elements[m][1]])
            class_values = np.array(class_values)
            class_mean = np.mean(class_values)
            csv_writer.writerow([LABELS[i], class_mean])
    return 0


files, filelabels = load_file()

# dataset stats
sum = 0
class_stats = dict.fromkeys(LABELS, 0)
for i in range(len(filelabels)):
    # find how many clips have each label
    for j in range(len(filelabels[i])):
        if filelabels[i][j] in class_stats.keys():
            class_stats[filelabels[i][j]] += 1
    # get the sum of all the lables to get average labels per clip
    sum += len(filelabels[i])
print('Class label stats:\n', class_stats)
print('Average class labels per file:', sum/len(filelabels))

feature_list = []
# Obtain MFCC for each audio clip
for i in range(len(files)):
    feature_list.append(features(files[i]))
feature_list = np.array(feature_list)
# Obtain similarity matrix for all the audios
similarity_matrix = similarity(feature_list)
# Get average similarity between each classes
class_similarity(similarity_matrix, filelabels)