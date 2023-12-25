import logging
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy import signal  # butterworth
from sklearn.decomposition import FastICA  # ICA
from sklearn import preprocessing  # scaling
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from models import EEGNet, Conformer


def load_eeg(subject: str) -> pd.DataFrame:
    """Load EEG raw data from certain path

    Parameters
    ----------
    subject : string
        Name of subject for specifying the path

    Returns
    -------
    eeg : pandas.DataFrame
        EEG raw data
    """
    eeg = pd.read_csv(
        f"./rec_data/{subject}_rec.csv",
        names=["unnamed", "idx", "left", "right", "cmd"],
        header=0,
    )
    eeg = eeg.drop(["unnamed", "idx"], axis=1)
    eeg["diff"] = eeg["left"] - eeg["right"]
    eeg = eeg.reindex(columns=["left", "right", "diff", "cmd"])
    return eeg


def plot_eeg(eeg: pd.DataFrame, col_names: list):
    """Plot EEG data

    Parameters
    ----------
    eeg : pandas.DataFrame
        EEG raw data
    col_names : list
        List of column names in order to plot certain column
    """
    plt.plot(eeg[col_names], label=col_names)
    plt.legend()
    plt.show()


def preprocess(
    eeg: pd.DataFrame, col_names: list, sample_freq: int = 1800, rec_sec: int = 4
) -> pd.DataFrame:
    """Returns preprocessed EEG data

    Parameters
    ----------
    eeg : pandas.DataFrame
        EEG raw data
    col_names : list
        List of column names in order to plot certain column
    sample_freq : int
        Frequency of sample EEG data
    rec_sec : int
        Recording seconds
    Returns
    -------
    eeg : pandas.DataFrame
        Preprocessed EEG data
    """

    # Butterworth filter
    sos = signal.butter(
        4, [4, 40], btype="bandpass", analog=False, output="sos", fs=sample_freq
    )
    for col in ["left", "right", "diff"]:
        col_dat = eeg.loc[:, [col]].to_numpy().reshape(-1)
        col_dat = signal.sosfiltfilt(sos, col_dat)
        eeg.loc[:, [col]] = col_dat

    # ICA
    ICA = FastICA(n_components=3, whiten="arbitrary-variance", random_state=0)
    X = eeg.loc[:, ["left", "right", "diff"]].to_numpy()
    X_trans = ICA.fit_transform(X)
    A_ = ICA.mixing_.T
    tmp = np.dot(X_trans, A_)
    for idx, col in enumerate(["left", "right", "diff"]):
        eeg.loc[:, [col]] = tmp[:, idx]

    # remove data across 3 sigma
    for col in col_names:
        time_series_data = eeg[col].copy()
        moving_average = time_series_data.rolling(window=sample_freq * rec_sec).mean()
        moving_std = time_series_data.rolling(window=sample_freq * rec_sec).std()
        outliers = np.abs(time_series_data - moving_average) > 3 * moving_std

        filtered_data = time_series_data
        for i in range(1, len(time_series_data)):
            if outliers[i]:
                filtered_data[i] = filtered_data[i - 1]

        eeg[col] = filtered_data

    # 0~1 Scaling
    mm = preprocessing.MinMaxScaler()
    eeg.loc[:, ["left", "right", "diff"]] = mm.fit_transform(
        eeg.loc[:, ["left", "right", "diff"]]
    )
    return eeg


def split_data(
    eeg: pd.DataFrame, infer_sec: int = 4, rec_sec: int = 8
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generates train/test dataset

    Parameters
    ----------
    eeg : pandas.DataFrame
        EEG data
    infer_sec : int, optional
        Seconds for inferring, by default 4
    rec_sec : int, optional
        Seconds each for each label, by default 8

    Returns
    -------
    df_train : numpy.ndarray
        EEG dataframe for training
    df_test : numpy.ndarray
        EEG dataframe for inferring
    infer_len : int
        Length of splitted EEG data
    """
    # splitting eeg-data by labels
    dataset = []
    split_idxs = [0]
    for i in range(1, len(eeg)):
        if eeg.iloc[i, 3] != eeg.iloc[i - 1, 3]:
            split_idxs.append(i)
    split_idxs.append(len(eeg))
    for i in range(len(split_idxs) - 1):
        tmp_data = eeg.iloc[split_idxs[i] : split_idxs[i + 1], :]
        dataset.append(tmp_data)

    # adjusting each data size to the minimum data size
    split_num = rec_sec // infer_sec
    infer_len = int(min([len(data) for data in dataset])) // split_num
    print(f"EEG length : {infer_len} frames")
    tmp_dataset = np.empty((1, infer_len, 4))
    for data in dataset:
        for idx in range(split_num):
            tmp_data = data.iloc[infer_len * idx : infer_len * (idx + 1)].to_numpy()
            tmp_data = np.expand_dims(tmp_data, 0)
            tmp_dataset = np.vstack((tmp_dataset, tmp_data))
    dataset = tmp_dataset[1:].astype("float32")

    # train,test split
    df_train, df_test = train_test_split(
        dataset, test_size=0.25, stratify=dataset[:, 0, 3]
    )

    return df_train, df_test, infer_len


class EEGDataset(Dataset):
    """Dataset of EEG for pytorch

    Attributes
    ----------
    dataset : numpy.ndarray
        Splitted EEG data

    Methods
    -------
    __getitem__(idx)
        Returns data specified by idx.
    __len__()
        Returns length of dataset.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, idx):
        label = self.dataset[idx, 0, -1]
        eeg = torch.tensor(self.dataset[idx, :, :-1])
        ret = {"eeg": eeg, "label": label}
        return ret

    def __len__(self):
        return len(self.dataset)


def trans_label(y: torch.Tensor, label_num: int = 3) -> torch.Tensor:
    """Transforms vector into one-hot matrix

    Parameters
    ----------
    y : torch.Tensor
        Label vector
    label_num : int
        Label number
    Returns
    -------
    label : torch.Tensor
        One-hot label, by default 3
    """
    label = torch.tensor(np.zeros((len(y), label_num)))
    for i in range(len(y)):
        label[i, int(y[i].item())] = 1
    return label


def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.modules.loss,
    optimizer: optim,
    device: torch.device,
    state: str = "Train",
) -> tuple[float, list, list]:
    """Trains the model

    Parameters
    ----------
    dataloader : torch.utils.data.dataloader.DataLoader
        EEG dataloader
    model : torch.nn.Module
        EEG model
    loss_fn : torch.nn.modules.loss
        Loss function
    optimizer : torch.optim
        Optimizer fuction
    device : torch.device
        Current device mode
    state : str, optional
        State either "Train" or "Val", by default "Train"

    Returns
    -------
    loss_sum : float
        Loss occured in latest train/infer
    pred_list : list
        List of predicted labels
    label_list : list
        List of true labels
    """
    loss_sum, correct = 0, 0
    pred_list, label_list = [], []
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    if state == "Train":
        model.train()
    elif state == "Val":
        model.eval()

    with torch.set_grad_enabled(state == "Train"):
        for batch, data in enumerate(dataloader):
            X = data["eeg"].to(device).unsqueeze(1).permute(0, 1, 3, 2)
            y = data["label"].to(device)
            y_one_hot = trans_label(y)

            pred = model(X)
            loss = loss_fn(pred, y_one_hot)
            loss_sum += loss.item()
            correct += (
                (torch.argmax(pred, dim=1) == torch.argmax(y_one_hot, dim=1))
                .type(torch.float)
                .sum()
                .item()
            )

            optimizer.zero_grad()
            if state == "Train":
                loss.backward()
                optimizer.step()

            preds = torch.argmax(pred, dim=1)
            pred_list += [data.item() for data in preds]
            label_list += [data.item() for data in y]

    loss_sum /= num_batches
    correct /= size / 100
    print(f"{state.ljust(5)}  Accuracy: {(correct):>5.1f}%, AvgLoss: {loss_sum:>7f}")
    return loss_sum, pred_list, label_list


def plot_cm(label: list, pred: list, state: str, save: bool, subject: str):
    """Plots confusion matrix

    Parameters
    ----------
    label : list
        List of true label
    pred : list
        List of predicted label
    state : string
        State either"Train" or "Val"
    save : bool
        Option to save the figure
    subject : string
        Name of the subject
    """
    cm = confusion_matrix(label, pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["jump", "dash", "stay"],
        yticklabels=["jump", "dash", "stay"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{state} Confusion Matrix")
    if save:
        plt.savefig(f"./output/{subject}_{state}_cf.png")
    plt.show()


def main():
    # exp parameters
    infer_sec = 4
    rec_sec = 8
    batch_size = 5
    num_epoch = 15
    channel_num = 3  # left, right, diff
    label_num = 3  # jump, dash, stay
    sample_freq = 1800

    # loading eeg
    subject = input("Subject Name: ")
    eeg = load_eeg(subject)
    print(f"EEG-data length : {len(eeg)}")
    col_names = ["left", "right", "diff"]
    plot_eeg(eeg, col_names)

    # preprocess
    eeg = preprocess(eeg, col_names, sample_freq, rec_sec)
    plot_eeg(eeg, col_names)

    # constructing dataloader
    df_train, df_test, infer_len = split_data(eeg, infer_sec, rec_sec)
    train_dataset = EEGDataset(df_train)
    test_dataset = EEGDataset(df_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model construction
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    args = sys.argv
    if len(args) == 2 and args[1].lower() != "eegnet":
        if args[1].lower() == "conformer":
            print("Using Conformer")
            model = Conformer(infer_len=infer_len, label=label_num, ch=channel_num)
    else:
        print("Using EEGNet")
        model = EEGNet(
            infer_len=infer_len, sample_freq=sample_freq, C=channel_num, N=label_num
        )

    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

    # model training
    train_loss_hist, test_loss_hist = [], []
    for t in range(num_epoch):
        string = "-" * 30
        print(f"Epoch {t+1}\n{string}")
        train_loss, train_pred, train_label = train(
            train_dataloader, model, loss_fn, optimizer, device, state="Train"
        )
        test_loss, test_pred, test_label = train(
            test_dataloader, model, loss_fn, optimizer, device, state="Val"
        )
        train_loss_hist.append(train_loss)
        test_loss_hist.append(test_loss)
        print()
    torch.save(model, f"./models/{subject}_model_weight.pth")

    plt.plot(train_loss_hist, label="train_loss")
    plt.plot(test_loss_hist, label="test_loss")
    plt.legend()
    plt.show()

    plot_cm(train_label, train_pred, state="Train", save=True, subject=subject)
    plot_cm(test_label, test_pred, state="Val", save=True, subject=subject)


if __name__ == "__main__":
    logging.getLogger("matplotlib.font_manager").disabled = True
    main()
