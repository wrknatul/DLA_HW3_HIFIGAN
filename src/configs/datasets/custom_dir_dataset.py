from pathlib import Path
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd

from src.datasets.base_dataset import BaseDataset

def read_file_to_string(filepath):
    """Reads a text file and returns its contents as a single string.
    Args:
        filepath: The path to the text file.
    Returns:
        The file content as a string, or None if an error occurs.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            file_content = file.read()
            return file_content
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
            "facebook/fastspeech2-en-ljspeech",
            arg_overrides={"vocoder": "hifigan", "fp16": False}
        )
        model = models[0]
        TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
        generator = task.build_generator([model], cfg)
        data = []
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".txt"]:
                entry["path"] = str(path)
                entry["text"] = read_file_to_string(entry["path"])
                sample = TTSHubInterface.get_model_input(task, entry["text"])
                _, entry["mels"] = TTSHubInterface.get_prediction(task, model, generator, sample)
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)