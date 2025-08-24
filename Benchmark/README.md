<a id="readme-top"></a>
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Unlicense License][license-shield]][license-url]

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This is the official repo for the paper: **PartImageNet++ Dataset: Enhancing Deep Learning Models with High-Quality Part Annotations**.

<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- GETTING STARTED -->
## Getting Started

### Environment for Segmentation

```bash
cd segmentation
conda create -n seg python=3.8
conda activate seg
pip install -r requirements.txt
```

### Environment for Few-shot Learning

For MetaBaseline: 
- Python 3.7.3
- Pytorch 1.2.0
- tensorboardX

For DeepEMD: 
- PyTorch >= version 1.1
- QPTH
- CVXPY
- OpenCV-python
- tensorboard

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Segmentation
First, you should put [annotations file](https://drive.google.com/drive/folders/10xHdH99vOs4tKBWWkFoK-hGah0Xa6EMq?usp=sharing) in `your_path/segmentation/PIN++/`.

Then, register the PIN++ dataset (see `your_path/segmentation/paco/data/datasets/builtin.py`).

If you want to train a Mask-RCNN with Resnet50 as backbone on PIN++, you can run the following cmd:
```bash
./tools/lazyconfig_train_net.py --config-file ./configs/PIN++_configs/r50_fpn.py --num-gpus 8
```
If you want to evaluate it, run:
```bash
./tools/lazyconfig_train_net.py --config-file ./configs/PIN++_configs/r50_fpn.py --eval-only --num-gpus 8 train.init_checkpoint=your_path_to_ckpt/model_final.pth
```
After that you can get the results of part segmentation and object segmentation. 
### Few-shot Learning

#### MetaBaseline
First, you should put the [pkl format files](https://drive.google.com/drive/folders/1nR8IFdypIg-FrRXiegH2lWGS_KzD6aRN?usp=sharing) of PIN++ in `your_path/fewshot/materials/pinpp/`.

Then, fill the path in `your_path/fewshot/configs/`.

If you want to train and test, run:
```bash
python train_meta.py --config configs/train_meta_pinpp.yaml
```

Then run:
```bash
python test_few_shot.py --shot 1
```

#### DeepEMD

First, you should change `DATA_DIR` in `train_pretrain.py` and `train_meta.py`.

If you want to train and test, run:
```bash
python train_pretrain.py -dataset pinpp -gpu 0,1,2,3
python train_meta.py -dataset pinpp -deepemd sampling -patch_list 9 -shot 1 -way 5 -solver opencv -gpu 0,1,2,3
```


<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- CONTRIBUTING -->
## Contributing

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the Unlicense License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Related Projects
- https://github.com/facebookresearch/paco

- https://github.com/icoz69/DeepEMD

- https://github.com/yinboc/few-shot-meta-baseline

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/LixiaoTHU/PartImageNetPP.svg?style=for-the-badge
[contributors-url]: https://github.com/LixiaoTHU/PartImageNetPP/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/LixiaoTHU/PartImageNetPP.svg?style=for-the-badge
[forks-url]: https://github.com/LixiaoTHU/PartImageNetPP/network/members
[stars-shield]: https://img.shields.io/github/stars/LixiaoTHU/PartImageNetPP.svg?style=for-the-badge
[stars-url]: https://github.com/LixiaoTHU/PartImageNetPP/stargazers
[issues-shield]: https://img.shields.io/github/issues/LixiaoTHU/PartImageNetPP.svg?style=for-the-badge
[issues-url]: https://github.com/LixiaoTHU/PartImageNetPP/issues
[license-shield]: https://img.shields.io/github/license/LixiaoTHU/PartImageNetPP.svg?style=for-the-badge
[license-url]: https://github.com/LixiaoTHU/PartImageNetPP/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/LixiaoTHU