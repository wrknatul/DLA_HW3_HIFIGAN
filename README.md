# PyTorch Template for DL projects

<p align="center">
  <a href="#about">About</a> •
  <a href="#tutorials">Tutorials</a> •
  <a href="#examples">Examples</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#useful-links">Useful Links</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
<a href="https://github.com/Blinorot/pytorch_project_template/generate">
  <img src="https://img.shields.io/badge/use%20this-template-green?logo=github">
</a>
<a href="https://github.com/Blinorot/pytorch_project_template/blob/main/LICENSE">
   <img src=https://img.shields.io/badge/license-MIT-blue.svg>
</a>
</p>

## About

This repository contains a HIFIGAN implementation.


## Installation

Installation may depend on your task. The general steps are the following:

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py
```

To run evaulation you need to install fairseq. I was installing by cloning git repository and resolving conflicts by this guide: [link](https://github.com/facebookresearch/audiocraft/issues/152#issuecomment-1877541684)



```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```
To change folder with audios or text use transcription_dir. To change output dir use audio_dir. To change checkpoint use from_pretrained. My cheeckpoint: [link](https://wandb.ai/hsee/HIFIGAN%20project/runs/ez5c1ei6/files/testing/checkpoint-epoch19.pth)

## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
