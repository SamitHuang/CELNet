# Evidence Localization for Pathology Images using Weakly Supervised Learning

This is a Pytorch implementation of CELNet described in the paper "Evidence Localization for Pathology Images using Weakly Supervised Learning" by Yongxiang Huang and Albert C.S. Chung.  

## Prerequisites
- `Python 3.6+`
- `Pytorch 1.2`

## Usage 
```
from celnet import celnet
model = celnet() 
...
logits = model(input_batch)
```
By default, the input image size is 96x96x3 and the number of class is 1 in this implementation. For applying it to other datasets with different image size and classes, please modify the code in `celnet.py` accordingly. 

## Reference 
If you find this code useful in your work, please cite:
```
@inproceedings{huang2019evidence,
  title={Evidence localization for pathology images using weakly supervised learning},
  author={Huang, Yongxiang and Chung, Albert CS},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={613--621},
  year={2019},
  organization={Springer}
}
```



