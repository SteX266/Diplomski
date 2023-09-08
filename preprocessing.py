import os
from pydub import AudioSegment


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt



def split_files(genres):
    i = 0
    for g in genres:
        j = -1
        print(f"{g}")
        for filename in os.listdir(os.path.join('./Data/genres_original', f"{g}")):

            song = os.path.join(f'./Data/genres_original/{g}', f'{filename}')
            j = j + 1
            for w in range(0, 10):
                i = i + 1
                # print(i)
                t1 = 3 * (w) * 1000
                t2 = 3 * (w + 1) * 1000
                newAudio = AudioSegment.from_wav(song)
                new = newAudio[t1:t2]
                new.export(f'./Data/genres_original_3sec/{g}/{g + str(j) + str(w)}.wav', format="wav")



def make_spectrogram(genres, batch_size=10):
    for g in genres:
        j = 0
        print(g)
        folder_path = os.path.join('./Data/genres_original_3sec', g)
        try:
            os.makedirs(os.path.join('./Data/spectrograms_3sec', g))
        except FileExistsError:
            pass  # If the folder already exists, continue

        files = os.listdir(folder_path)
        num_files = len(files)
        for i in range(0, num_files, batch_size):
            batch_files = files[i:i+batch_size]
            for filename in batch_files:
                j += 1
                song = os.path.join(folder_path, filename)
                try:
                    y, sr = librosa.load(song, duration=3)
                    mels = librosa.feature.melspectrogram(y=y, sr=sr)
                    fig, ax = plt.subplots()
                    librosa.display.specshow(librosa.power_to_db(mels, ref=np.max), ax=ax)
                    plt.savefig(f'./Data/spectrograms_3sec/{g}/{g}{j}.png')
                    plt.close(fig)
                except Exception as e:
                    print(f"Error processing {song}: {e}")


if __name__ == '__main__':
    genres = 'blues classical country disco pop hiphop metal reggae rock jazz'
    genres = genres.split()
#    split_files(genres)
    make_spectrogram(genres)