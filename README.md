# MaskStyleTransfer

![](teaser.gif)

## Prerequisites
- [Pytorch](http://pytorch.org/)
- [torchvision](https://github.com/pytorch/vision)
### Optional
- [kaggle-cli](https://github.com/floydwch/kaggle-cli) : Used to download WikiArt by command line

## DATASET
### MSCOCO
```
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
```
### WikiArt
- Either manually download from [kaggle](https://www.kaggle.com/c/painter-by-numbers).
- Or install kaggle-cli and download by running:
```
kg download -u <username> -p <password> -c painter-by-numbers -f train.zip
```

## Training
### Training with loss network
```
python Train.py --vgg_dir PATH_TO_VGG --decoder_dir PATH_TO_DECODER --layer r31 --contentPath PATH_TO_MSCOCO --stylePath PATH_TO_WikiArt --reg_weight 100 --outf OUTPUT_DIR
```

### Training with style loss calculated on transferred feature
```
python TrainLossonEncoder.py --vgg_dir PATH_TO_MODELS/vgg_normalised_conv3_1.t7 --vgg4_dir PATH_TO_MODELS/vgg_normalised_conv4_1.t7 --decoder_dir PATH_TO_MODELS/feature_invertor_conv3_1.t7 --layer r31 --contentPath PATH_TO_MSCOCO --stylePath PATH_TO_WikiArt --outf OUTPUT_DIR
```
Intermediate results and weight will be stored in `OUTPUT_DIR`

# Testing

## Test without Mask
```
python TestWithoutMask.py
```

## Test with Mask
```
python TestWithMask.py
```

## Testing on Video
```
python VideoTransfer.py
```
