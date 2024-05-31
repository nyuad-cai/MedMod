# 🩻 MedMod: A Multimodal Benchmark for Clinical Prediction Tasks with Electronic Health Records and Chest X-Ray Images

*Clinical prediction with multimodal learning using Electronic Health Records and Chest X-Ray images *

**MedMod: A Multimodal Benchmark for Clinical Prediction Tasks with Electronic Health Records and Chest X-Ray Image** (NeurIPS 2024 Datasets and Benchmarks Track, **Paper**) [[Paper](https://clinicalailab.com/)]  <br>

## Release

- [06/05] 🩻 **MedMod** (Initial) models and clinical tasks are released, [[Code](https://github.com/nyuad-cai/Multimodal-BenchMark/)] 

- [2024/06/05] [MedMod](https://clinicalailab.com/) is submitted to NeurIPS 2024 Datasets and Benchmarks Track as **workshop paper**.

<!-- <a href="https://llava.hliu.cc/"><img src="assets/demo.gif" width="70%"></a> -->

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
**Usage and License Notices**: This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses. This project does not impose any additional constraints beyond those stipulated in the original licenses. Furthermore, users are reminded to ensure that their use of the dataset and checkpoints is in compliance with all applicable laws and regulations.


## Contents
- [Install](#install)
- [Demo](#Demo)
- [Models](https://github.com/nyuad-cai/Multimodal-BenchMark/blob/main/docs/MODEL_ZOO.md)
- [Datasets](https://github.com/nyuad-cai/Multimodal-BenchMark/blob/main/docs/Data.md)
- [Train](#train)
- [Evaluation](#evaluation)

## Install

1. Clone this repository and navigate to MedMod folder
```bash
git clone https://github.com/nyuad-cai/Multimodal-BenchMark
cd MedMod
```

2. Install Package
```Shell
conda create -n medmod python=3.10 -y
conda activate medmod
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```
### Upgrade to latest code base

```Shell
git pull
pip install -e .

# if you see some import errors when you upgrade,
# please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir
```

## Model Weights
Please check out our [Models](https://github.com/nyuad-cai/Multimodal-BenchMark/blob/main/docs/MODEL_ZOO.md) for all public MedMod checkpoints, and the instructions of how to use the weights.

## Demo

## Evaluation

## Citation

If you find MedMod useful for your research and applications, please cite using this BibTeX:
```bibtex
@misc{author2024medmod,
    title={MedMod: A Multimodal Benchmark for Clinical Prediction Tasks with Electronic Health Records and Chest X-Ray Images},
    url={https://github.com/nyuad-cai/Multimodal-BenchMark},
    author={All Clinical AI Lab woo},
    month={June},
    year={2024}
}
```

## Acknowledgement


## Related Projects

- [MedFuse: Multi-modal fusion with clinical time-series data and chest X-ray images](https://arxiv.org/pdf/2207.07027)

- MedFuse [Code](https://github.com/nyuad-cai/MedFuse/tree/main)
