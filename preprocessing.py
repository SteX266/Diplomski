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

def split_files_2sec(genres):
    i = 0
    for g in genres:
        j = -1
        print(f"{g}")
        for filename in os.listdir(os.path.join('./Data/genres_original', f"{g}")):
            song = os.path.join(f'./Data/genres_original/{g}', f'{filename}')
            j += 1
            for w in range(0, 15):  # Change the range to split into 2-second segments
                i += 1
                t1 = 2 * (w) * 1000
                t2 = 2 * (w + 1) * 1000
                newAudio = AudioSegment.from_wav(song)
                new = newAudio[t1:t2]
                new.export(f'./Data/genres_original_2sec/{g}/{g}{j:02d}{w:02d}.wav', format="wav")

def split_files_5sec(genres):
    i = 0
    for g in genres:
        j = -1
        print(f"{g}")
        for filename in os.listdir(os.path.join('./Data/genres_original', f"{g}")):
            song = os.path.join(f'./Data/genres_original/{g}', f'{filename}')
            j += 1
            for w in range(0, 6):  # Change the range to split into 5-second segments
                i += 1
                t1 = 5 * (w) * 1000
                t2 = 5 * (w + 1) * 1000
                newAudio = AudioSegment.from_wav(song)
                new = newAudio[t1:t2]
                new.export(f'./Data/genres_original_5sec/{g}/{g}{j:02d}{w:02d}.wav', format="wav")




def make_spectrogram(genres, duration=3):
    for g in genres:
        j = 0
        print(g)
        folder_path = os.path.join(f'./Data/genres_original_{duration}sec', g)
        try:
            os.makedirs(os.path.join(f'./Data/spectrograms_{duration}sec', g))
        except FileExistsError:
            pass  # If the folder already exists, continue

        files = os.listdir(folder_path)
        for filename in files:
            j += 1
            song = os.path.join(folder_path, filename)
            try:
                y, sr = librosa.load(song, duration=duration)
                mels = librosa.feature.melspectrogram(y=y, sr=sr)
                fig, ax = plt.subplots()
                librosa.display.specshow(librosa.power_to_db(mels, ref=np.max), ax=ax)
                plt.savefig(f'./Data/spectrograms_{duration}sec/{g}/{g}{j}.png')
                plt.close(fig)
            except Exception as e:
                print(f"Error processing {song}: {e}")


if __name__ == '__main__':
    genres = 'blues classical country disco pop hiphop metal reggae rock jazz'
    genres = genres.split()
 #   split_files_5sec(genres)
    make_spectrogram(genres,30)