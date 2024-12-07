import torch
from torch.nn.utils.rnn import pad_sequence
from src.datasets.mel_generator import MelSpectrogram, MelSpectrogramConfig


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    audios  = []
    audio_lens = []
    texts = []
    paths = []
    for item in dataset_items:
        audios.append(item["data_object"].squeeze(0))
        audio_lens.append(item["audio_len"])
        texts.append(item["text"])
        paths.append(item["data_path"])
    
    mel_generator = MelSpectrogram(MelSpectrogramConfig())
    final_wavs = pad_sequence(audios, batch_first=True)
    mels = mel_generator(final_wavs)
    return {
        "wavs": final_wavs.unsqueeze(1),
        "mels": mels,
        "audio_lens": torch.Tensor(audio_lens),
        "text": texts,
        "paths": paths
    }
