# Predicting performance difficulty from piano sheet music images

Code of the paper P. Ramoneda, J. J. Valero-Mas, D. Jeong & X. Serra, Predicting performance difficulty from piano sheet music images, in Proc. of the 24th Int. Society for Music Information Retrieval Conf., Milan, Italy (2023).


To cite this work, please use the following bibtex entry:

```
@inproceedings{ramoneda2023predicting,
  title={Predicting performance difficulty from piano sheet music images},
  author={Ramoneda, P. and Valero-Mas, J. J. and Jeong, D. and Serra, X.},
  booktitle={Proc. of the 24th Int. Society for Music Information Retrieval Conf.},
  year={2023},
  address={Milan, Italy}
}
```

 [Paper](https://arxiv.org/pdf/2309.16287.pdf) | [Dataset](https://zenodo.com/record/8126801) | [Demo](https://musiccritic.upf.edu/pdf_difficulty/) 

## Abstract

Estimating the performance difficulty of a musical score
is crucial in music education for adequately designing the
learning curriculum of the students. Although the Music
Information Retrieval community has recently shown interest in this task, existing approaches mainly use machine-
readable scores, leaving the broader case of sheet music
images unaddressed. Based on previous works involving sheet music images, we use a mid-level representa-
tion, bootleg score, describing notehead positions relative
to staff lines coupled with a transformer model. This architecture is adapted to our task by introducing an encoding
scheme that reduces the encoded sequence length to oneeighth of the original size. In terms of evaluation, we con-
sider five datasets—more than 7500 scores with up to 9 difficulty levels—, two of them particularly compiled for this
work. The results obtained when pretraining the scheme
on the IMSLP corpus and fine-tuning it on the considered
datasets prove the proposal’s validity, achieving the bestperforming model with a balanced accuracy of 40.34% and
a mean square error of 1.33. Finally, we provide access
to our code, data, and models for transparency and reproducibility.

## System-level dependencies

Please ensure the following dependencies are installed on your system:

For Debian/Ubuntu based systems:

```sh
sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6 imagemagick ghostscript -y
```

## Python dependencies

```sh
python -m pip install -r requirements.txt
```

## Inference

```Python
from pdf_difficulty.predict_difficulty import predict_difficulty

diff_cipi, diff_ps, diff_fs = predict_difficulty("examples/124.pdf")
print(diff_cipi, diff_ps, diff_fs)
```