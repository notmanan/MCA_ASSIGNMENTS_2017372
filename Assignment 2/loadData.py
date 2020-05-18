from scipy.io import wavfile
import glob
import pickle

names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
data = []
for i in range(len(names)):
    files = glob.glob("validation\\" + names[i] + "\\*")
    folderdata = []
    for j in range(len(files)):
        sr, d = wavfile.read(files[j])
        folderdata.append([sr, d])
    pickle.dump(folderdata, open(names[i]+"validationDataFile.p", "wb"))
    data.append(folderdata)
