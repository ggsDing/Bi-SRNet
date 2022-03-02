# Bi-SRNet
Pytorch codes of 'Bi-Temporal Semantic Reasoning for the Semantic Change Detection in HR Remote Sensing Images' [[paper]](https://ieeexplore.ieee.org/document/9721305)


<img src="https://github.com/ggsDing/Bi-SRNet/blob/main/FlowChart.png">

<img src="https://github.com/ggsDing/Bi-SRNet/blob/main/SR.png" height="200"> <img src="https://github.com/ggsDing/Bi-SRNet/blob/main/BiSR.png" height="200">
<img src="https://github.com/ggsDing/Bi-SRNet/blob/main/SCLoss.png" height="200">

**Data preparation**
1. Split the SECOND data into training, validation (and testing if available) set and organize them as follows:
-YOUR_DATA_DIR
  -Train
    -image
    -labels
  -Val
    -image
    -labels
2. Set the data path as YOUR_DATA_DIR at *dataset -RS_ST.py*

If you find our work useful or interesting, please consider to cite:
> Ding L, Guo H, Liu S, et al. Bi-temporal semantic reasoning for the semantic change detection in hr remote sensing images[J]. IEEE Transactions on Geoscience and Remote Sensing, 2022.
