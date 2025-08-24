import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from math import atanh

"""
[dataframe]
subject	
run	
char	    single char level
onset	    time
embedding	vector based on 'Whisper'  (np.ndarray, shape=(d,))
EEG	        slides (np.ndarray, shape=(n_channels, n_timepoints))
"""


# the same as function of encoding_model
def encoding_model(eeg_data, embedding, split_ratio=0.8):
    """
    eeg_data: (n_tokens, n_channels, n_timepoints)
    embedding: (n_tokens, d)
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import train_test_split

    n_tokens, n_channels, n_timepoints = eeg_data.shape
    assert embedding.shape[0] == n_tokens, "Mismatch in token number"

    X_train, X_test, y_train, y_test = train_test_split(
        embedding, eeg_data, test_size=1-split_ratio, random_state=42
    )

    alphas = np.logspace(-3, 3, 7)
    channel_correlations = np.zeros(n_channels)

    for channel_idx in range(n_channels):
        y_train_channel = y_train[:, channel_idx, :]
        y_test_channel = y_test[:, channel_idx, :]

        # flatten timepoints for regression
        y_train_flat = y_train_channel.reshape(y_train_channel.shape[0], -1)
        y_test_flat = y_test_channel.reshape(y_test_channel.shape[0], -1)

        ridge = RidgeCV(alphas=alphas, store_cv_results=True)
        ridge.fit(X_train, y_train_flat)

        y_pred_flat = ridge.predict(X_test)
        y_pred = y_pred_flat.reshape(y_test_channel.shape)

        time_correlations = []
        for time_idx in range(n_timepoints):
            r, _ = pearsonr(y_test_channel[:, time_idx], y_pred[:, time_idx])
            time_correlations.append(r)
        mean_correlation = np.mean(time_correlations)
        if abs(mean_correlation) >= 0.999:
            mean_correlation = np.sign(mean_correlation) * 0.999
        fisher_z = atanh(mean_correlation)

        channel_correlations[channel_idx] = fisher_z
    return channel_correlations


# multi char EEG and embedding
def build_word_level_eeg_from_onset(char_df, word_length=2, eeg_mode="concat"):
    """
    char_df:  ["subject", "run", "char", "onset", "embedding", "EEG"]
             - embedding: np.ndarray, (d,)
             - EEG: np.ndarray, (n_channels, n_timepoints)
    word_length: 2=2char, 4=4char
    eeg_mode: "concat" or "mean"
    """
    embeddings, eegs = [], []

    for (subj, run), run_df in char_df.groupby(["subject", "run"]):
        run_df = run_df.sort_values("onset").reset_index(drop=True)
        n = len(run_df)

        for i in range(n - word_length + 1):
            group = run_df.iloc[i:i+word_length]

            # 
            # if not np.all(np.diff(group["onset"]) < 0.5):  
            #    continue

            # average
            emb = np.mean(np.stack(group["embedding"].to_numpy()), axis=0)

            # EEG combination
            eeg_list = list(group["EEG"].to_numpy())
            if eeg_mode == "concat":
                eeg = np.concatenate(eeg_list, axis=-1)  # (n_channels, n_timepoints*word_length)
            elif eeg_mode == "mean":
                eeg = np.mean(np.stack(eeg_list), axis=0)  # (n_channels, n_timepoints)
            else:
                raise ValueError("eeg_mode must be concat or mean")

            embeddings.append(emb)
            eegs.append(eeg)

    embeddings = np.stack(embeddings)
    eegs = np.stack(eegs)
    return eegs, embeddings


# single char vs 2-char vs 4-char
def run_word_vs_char(char_df, split_ratio=0.8):
    results = {}
    eegs_char = np.stack(char_df["EEG"].to_numpy())
    embs_char = np.stack(char_df["embedding"].to_numpy())
    results["char"] = encoding_model(eegs_char, embs_char, split_ratio)

    eegs_2, embs_2 = build_word_level_eeg_from_onset(char_df, word_length=2)
    results["2-char"] = encoding_model(eegs_2, embs_2, split_ratio)

    eegs_4, embs_4 = build_word_level_eeg_from_onset(char_df, word_length=4)
    results["4-char"] = encoding_model(eegs_4, embs_4, split_ratio)

    return results

# Contrast Visualization
def plot_word_vs_char(results):
    plt.figure(figsize=(8, 5))
    labels = []
    mean_r2 = []
    for key, corr in results.items():
        r2 = np.mean(corr ** 2)  
        labels.append(key)
        mean_r2.append(r2)

    plt.bar(labels, mean_r2, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.ylabel("Mean $R^2$")
    plt.title("Decoding performance: Char vs 2-char vs 4-char")
    plt.show()


results = run_word_vs_char(char_df, split_ratio=0.8)
plot_word_vs_char(results)
