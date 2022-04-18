# Bi-SRNet
Pytorch codes of 'Bi-Temporal Semantic Reasoning for the Semantic Change Detection in HR Remote Sensing Images' [[paper]](https://ieeexplore.ieee.org/document/9721305)


<p align="center">
<img src="https://github.com/ggsDing/Bi-SRNet/blob/main/FlowChart.png">

<img src="https://github.com/ggsDing/Bi-SRNet/blob/main/BiSR.png" width="500">
<img src="https://github.com/ggsDing/Bi-SRNet/blob/main/SCLoss.png" width="600">
</p>

**Data preparation:**
1. Split the SCD data into training, validation and testing (if available) set and organize them as follows:

>YOUR_DATA_DIR
>  - Train
>    - im1
>    - im2
>    - label1
>    - label2
>  - Val
>    - im1
>    - im2
>    - label1
>    - label2
>  - Test
>    - im1
>    - im2
>    - label1
>    - label2
    
2. Find *-datasets -RS_ST.py*, set the data root in *Line 22* as *YOUR_DATA_DIR*

**Reference**

If you find our work useful or interesting, please consider to cite:
> Ding L, Guo H, Liu S, et al. Bi-temporal semantic reasoning for the semantic change detection in hr remote sensing images[J]. IEEE Transactions on Geoscience and Remote Sensing, 2022.
