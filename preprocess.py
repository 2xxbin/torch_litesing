from scipy.signal import resample_poly
import soundfile as sf
from tqdm import tqdm
import pyworld as pw
import numpy as np
import tempfile
import librosa
import pysptk
import random
import os

# --------- setting -----------
DATA_FOLDER_NAME = "ritsu"
REST_NOTES = ["pau", 'AP', "sil", "SP"]

SR = 32000
HOP_TIME_MS = 5
HOP_SIZE = int(SR * HOP_TIME_MS / 1000)
MGC_DIM = 60
BAP_DIM = 4
ALPHA = 0.455
FFT_LEN = 1024
F0_MIN = 40
F0_MAX = 1100
PREPROCESS_FOLDER_LIST = ['mgc','bap','f0','vuv','phone','note', 'energy']

current_folder = os.path.dirname(__file__)
raw_folder = os.path.join(current_folder, 'data', "raw", DATA_FOLDER_NAME)
preprocessed_folder = os.path.join(current_folder, 'data', 'preprocessed', DATA_FOLDER_NAME)
wav_folder = os.path.join(raw_folder, 'wav')
lab_folder = os.path.join(raw_folder, 'lab')
train_folder = os.path.join(preprocessed_folder, 'train')
validation_folder = os.path.join(preprocessed_folder, "validation")

all_phone = []

def ensure_dirs(*dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

def parse_lab(lab_path):
    lab = []
    with open(lab_path, encoding='utf-8') as f:
        for line in f:
            s, e, ph = line.strip().split()
            s_sec = float(s) / 10_000_000
            e_sec = float(e) / 10_000_000
            lab.append((s_sec, e_sec, ph))
    return lab

def split_wav_and_lab(wav_path, lab_path, out_wav_dir, out_lab_dir, chunk_sec=10):
    x, sr = sf.read(wav_path)
    total_sec = len(x) / sr
    lab = parse_lab(lab_path)

    chunk_idx = 0
    start_time = 0.0
    while start_time < total_sec:
        end_time = min(start_time + chunk_sec, total_sec)

        start_idx = int(start_time * sr)
        end_idx = int(end_time * sr)
        chunk_audio = x[start_idx:end_idx]
        chunk_wav_path = os.path.join(out_wav_dir, f"{os.path.splitext(os.path.basename(wav_path))[0]}_{chunk_idx:03d}.wav")
        sf.write(chunk_wav_path, chunk_audio, sr)

        chunk_lab = []
        for s, e, ph in lab:
            if e <= start_time: continue
            if s >= end_time: continue
            chunk_s = max(s, start_time) - start_time
            chunk_e = min(e, end_time) - start_time
            chunk_lab.append([int(chunk_s * 10_000_000), int(chunk_e * 10_000_000), ph])
        chunk_lab_path = os.path.join(out_lab_dir, f"{os.path.splitext(os.path.basename(lab_path))[0]}_{chunk_idx:03d}.lab")
        with open(chunk_lab_path, "w", encoding='utf-8') as f:
            for s, e, ph in chunk_lab:
                f.write(f"{s} {e} {ph}\n")
        start_time += chunk_sec
        chunk_idx += 1

def extract_features(x, fs):
    if fs != SR:
        x = resample_poly(x, up=SR, down=fs)
        fs = SR
    f0, t = pw.harvest(x, fs, frame_period=HOP_TIME_MS, f0_floor=F0_MIN, f0_ceil=F0_MAX)
    sp = pw.cheaptrick(x, f0, t, fs, fft_size=FFT_LEN)
    ap = pw.d4c(x, f0, t, fs, fft_size=FFT_LEN)
    mgc = pysptk.sp2mc(sp, order=MGC_DIM-1, alpha=ALPHA)
    bap = pw.code_aperiodicity(ap, fs)
    if bap.shape[1] > BAP_DIM:
        bap = bap[:, :BAP_DIM]
    vuv = (f0 > 0).astype(np.float32)
    logf0 = np.log(f0 + 1e-8)
    logf0[f0 == 0] = 0
    hop_length = int(SR * HOP_TIME_MS / 1000)
    rms = librosa.feature.rms(y=x, frame_length=FFT_LEN, hop_length=hop_length)
    rms = rms.squeeze()
    return mgc, bap, logf0, vuv, f0, rms

def extract_midi_and_dur_framewise(lab, midi_pitch, time_step, num_frames):
    # 프레임 단위 라벨 시퀀스 생성 (T, )
    midi_seq = np.full((num_frames,), -1, dtype=np.int32)
    phone_seq = np.zeros((num_frames,), dtype=np.int8)
    for s_sec, e_sec, ph in lab:
        s_idx = int(np.floor(s_sec / time_step))
        e_idx = int(np.ceil(e_sec / time_step))
        if e_idx > num_frames:
            e_idx = num_frames
        if s_idx >= e_idx:
            continue

        if not ph in all_phone: all_phone.append(ph)
        ph_idx = all_phone.index(ph)
        phone_seq[s_idx:e_idx] = ph_idx
        if ph in REST_NOTES:
            midi_seq[s_idx:e_idx] = -1
        else:
            cur_pitch = midi_pitch[s_idx:e_idx]
            cur_pitch = cur_pitch[cur_pitch > 0]
            if cur_pitch.size > 0:
                counts = np.bincount(np.round(cur_pitch).astype(np.int64))
                midi_num = counts.argmax()
                midi_seq[s_idx:e_idx] = midi_num
            else:
                midi_seq[s_idx:e_idx] = -1
    return midi_seq.astype(np.float32), phone_seq.astype(np.float32)

def save_features(out_folder_dict, basename, mgc, bap, logf0, vuv, phone_seq, midi_seq, energy):
    np.save(os.path.join(out_folder_dict['mgc'],  basename + "_mgc.npy"), mgc)
    np.save(os.path.join(out_folder_dict['bap'],  basename + "_bap.npy"), bap)
    np.save(os.path.join(out_folder_dict['f0'],   basename + "_f0.npy"), logf0)
    np.save(os.path.join(out_folder_dict['vuv'],  basename + "_vuv.npy"), vuv)
    np.save(os.path.join(out_folder_dict['phone'],basename + "_phone.npy"), phone_seq)
    np.save(os.path.join(out_folder_dict['note'], basename + "_note.npy"), midi_seq)
    np.save(os.path.join(out_folder_dict['energy'], basename + "_energy.npy"), energy)

def process_and_save(wav_list, temp_wav_dir, temp_lab_dir, out_folder_dict):
    for wav_name in tqdm(wav_list):
        wav_path = os.path.join(temp_wav_dir, wav_name)
        basename = os.path.splitext(wav_name)[0]
        lab_path = os.path.join(temp_lab_dir, basename + ".lab")

        x, fs = sf.read(wav_path)
        mgc, bap, logf0, vuv, rawf0, energy = extract_features(x, fs)

        midi_pitch = np.zeros_like(rawf0)
        midi_pitch[rawf0 > 0] = librosa.hz_to_midi(rawf0[rawf0 > 0])
        time_step = HOP_TIME_MS / 1000

        lab = parse_lab(lab_path)
        num_frames = mgc.shape[0]
        midi_seq, phone_seq = extract_midi_and_dur_framewise(lab, midi_pitch, time_step, num_frames)

        save_features(out_folder_dict, basename, mgc, bap, logf0, vuv, phone_seq, midi_seq, energy)

def main():
    with tempfile.TemporaryDirectory() as tempdir:
        temp_wav_dir = os.path.join(tempdir, "wav")
        temp_lab_dir = os.path.join(tempdir, "lab")
        ensure_dirs(tempdir, temp_wav_dir, temp_lab_dir)

        train_dirs = {k: os.path.join(train_folder, k) for k in PREPROCESS_FOLDER_LIST}
        val_dirs   = {k: os.path.join(validation_folder, k) for k in PREPROCESS_FOLDER_LIST}
        ensure_dirs(preprocessed_folder, train_folder, validation_folder, *train_dirs.values(), *val_dirs.values())

        # 1. split audio & lab
        wav_list = [file for file in os.listdir(wav_folder) if file.endswith('.wav')]
        print("audio split..")
        for wav_name in tqdm(wav_list):
            wav_path = os.path.join(wav_folder, wav_name)
            basename = os.path.splitext(wav_name)[0]
            lab_path = os.path.join(lab_folder, basename + ".lab")
            split_wav_and_lab(wav_path, lab_path, temp_wav_dir, temp_lab_dir, chunk_sec=10)

        # 2. train/validation split
        all_split_wav = [file for file in os.listdir(temp_wav_dir) if file.endswith('.wav')]
        random.shuffle(all_split_wav)
        val_wavs = all_split_wav[:10]
        train_wavs = all_split_wav[10:]

        # 3. feature extraction & save
        print("save train data features...")
        process_and_save(train_wavs, temp_wav_dir, temp_lab_dir, train_dirs)
        print("save validation data features...")
        process_and_save(val_wavs, temp_wav_dir, temp_lab_dir, val_dirs)

        print("save all phoneme list...")
        np.save(os.path.join(preprocessed_folder, "all_phone.npy"), all_phone)

if __name__ == "__main__":
    main()