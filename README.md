# Modality Divergence-aware Dynamic Gating for RGB-T Semantic Segmentation
This is official pytorch implementation of "Modality Divergence-aware Dynamic Gating for RGB-T Semantic Segmentation".  <br/>

## Introduction
<div align="center">
<img src="framework.png" width="850"/>
</div>
<br/>RGB-Thermal semantic segmentation employs complementary visual information of both RGB and thermal images to predict pixel-level label maps. How to learn their complementary features and fuse them poses great challenge. However, previous methods adopt the same fusion strategy on all features, which neglects two facts: 1) non-symmetry of RGB and thermal modalities makes semantic-consistency and semantic-inconsistency features co-exist between two modalities, and 2) low-level feature fusion may bring about some noise such as clutter background which adds difficulty in feature decoding. To address these issues, we develop a Modality Divergence-aware Dynamic Gating approach for RGB-T semantic segmentation. In particular, it primarily consists of the modality divergence interaction module and the semantic dynamic decoding module. The former focuses on semantic divergence features by adopting local cross-attention to capture complementary features of corresponding locations and using bi-directional scanning of channel Mamba to establish dual-modality feature relations. Meanwhile, the latter involves dynamic linear layer to model interaction relations between high-level and low-level features by dynamic weighting model, while developing a dynamic frequency gating mechanism to strengthen or weaken features in frequency domain. This helps segment small objects and better capture object contours. Empirical studies on three benchmarks including MFNet, PST900, and FMB demonstrate the superiority of the proposed approach.

## Install Dependencies
The code is written in Python 3.10 and Cuda 11.8 using the following libraries:

```
python==3.10.14
torch==2.3.1
torchaudio==2.3.1
torchvision==0.18.1
natten==0.17.1
```

Install the libraries using [requirements.txt](requirements.txt) as:

```
pip install -r requirements.txt
```

## Data
For training, download the MFNet dataset from [here](https://github.com/haqishen/MFNet-pytorch) and please modify the path to the dataset in code.
<br/>


## Folder Structure
While training, the models are saved in a folder specifying the hyper-parameters for that run under the [exp](exp) directory, including the .pth file and the .log file. The directory structure looks like this:
```
TriKD_SemiSeg
│
├─configs
│      cityscapes_triple11m_512_e300.yaml
│      cityscapes_triple21m_512_e300.yaml
│      eval.yaml
│
├─dataset
│      semi.py
│      transform.py
│
├─exp
│      exp_log.txt
│
├─model
│  │  helpers.py
│  │  resnet101.pth
│  │  resnet50.pth
│  │
│  ├─backbone
│  │      resnet.py
│  │      tinyvit_kd.py
│  │      vit_kd.py
│  │      xception.py
│  │
│  └─semseg
│          decoder.py
│          deeplabv3plus.py
│          fft_attn.py
│          model_helper_kd.py
│          neck.py
│
├─pretrained
│      pretrained model.txt
│
├─scripts
│      eval.sh
│      train_TriKD_autocast.sh
│
├─splits
│  └─cityscapes
│      │  eval.txt
│      │  val.txt
│      │
│      ├─1_16
│      │      labeled.txt
│      │      unlabeled.txt
│      │
│      ├─1_2
│      │      labeled.txt
│      │      unlabeled.txt
│      │
│      ├─1_30
│      │      labeled.txt
│      │      unlabeled.txt
│      │
│      ├─1_4
│      │      labeled.txt
│      │      unlabeled.txt
│      │
│      └─1_8
│              labeled.txt
│              unlabeled.txt
│
├─util
│        classes.py
│        dist_helper.py
│        eval_helper.py
│        ohem.py
│        utils.py
│  eval.py
│  readme.md
│  requirements.txt
│  semi_TriKD_autocast.py
└─ TriKD.png
```

<br/>

## Quantitative results on Cityscapes
<table>
   <tr>
      <td>Model</td>
      <td>1/16</td>
      <td>1/8</td>
      <td>1/4</td>
      <td>1/2</td> 
   </tr>
   <tr>
      <td>TriVN 11M</td>
      <td>69.18</td>
      <td>73.25</td>
      <td>75.02</td>
      <td>75.98</td>
   </tr>
   <tr>
      <td>TriVN 21M</td>
      <td>72.70</td>
      <td>76.44</td>
      <td>78.01</td>
      <td>79.12</td>
   </tr>
</table>


## Qualitative results  on Cityscapes
![Image](mf_vis.png)<br/>


## Training & Testing
### Training the MiT-B2 on MF Datasets:
Use the command below for training, modify the run-time arguments (like hyper-parameters for training, path to save the models, etc.) as required:
```
bash tools/train_ddp.sh 
```
<br/>

### Testing the MiT-B2 on MF Datasets:
Before testing, place the trained model file and modify model_path='your model file path'in the scripts/eval.sh and use the command below for testing:
```
python tools/val_mm.py 
```
<br/>

## Citation

If you find this repo useful, please cite the following paper.

```
@article{Modality Divergence-aware Dynamic Gating for RGB-T semantic segmentation,
  author    = {},
  title     = {},
  journal   = {},
  volume    = {},
  pages     = {},
  year      = {2025}
}
```

## Contact

For any issues, please contact Mr. Tang Chen via email tangc@hdu.edu.cn

