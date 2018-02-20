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

## Train with Matting Loss
```
python TrainWithMatting.py --vgg_dir PATH_TO_VGG --decoder_dir PATH_TO_DECODER --contentPath PATH_TO_MSCOCO --stylePath PATH_TO_WikiArt --fineSize 256 --finetune weights/r31.pth
```
Note: if use `--finetune weights/r31.pth`, then we fine tune the model based on previous trained transfer model

## Training AutoEncoder

### Train Decoder with Compress and Unzip layers

```
python TrainAE.py --vgg_dir PATH_TO_MODELS/vgg_normalised_conv3_1.t7 --decoder_dir PATH_TO_MODELS/feature_invertor_conv3_1.t7 --stylePath PATH_TO_WikiArt --contentPath PATH_TO_MSCOCO --layer r31
```
- Note: For ReLU4, change corresponding weight path and use `--layer r41`

### Test AutoEncoder

```
python TestAE.py --vgg_dir PATH_TO_MODELS/vgg_normalised_conv3_1.t7 --decoder_dir PATH_TO_MODELS/feature_invertor_conv3_1.t7 --mode withCU
```
- NOTE: mode determines whether test with compress layer or not. To not include compress layer, use `--mode withoutCU`.

## Train Style Transfer Model

```
python Train.py --vgg_dir PATH_TO_VGG --decoder_dir PATH_TO_DECODER --layer r31 --contentPath PATH_TO_MSCOCO --stylePath PATH_TO_WikiArt --reg_weight 10 --outf OUTPUT_DIR --compress_dir PRE_TRAINED_COMPRESS_LAYER_PATH --unzip_dir PRE_TRAINED_UNZIP_LAYER_PATH --new_decoder_dir FINETUNED_DECODER_PATH
```
Intermediate results and weight will be stored in `OUTPUT_DIR`

# Testing

## Test without Mask
```
python TestWithoutMask.py --vgg_dir PATH_TO_VGG --decoder_dir PATH_TO_DECODER --layer r31 --outf OUTPUT_DIR --compress_dir PRE_TRAINED_COMPRESS_LAYER_PATH --unzip_dir PRE_TRAINED_UNZIP_LAYER_PATH --new_decoder_dir FINETUNED_DECODER_PATH --matrixPath PATH_TO_PRE_TRAINED_MODEL
```

## Test with Mask
```
python TestWithMask.py
```

## Testing on Video
```
python VideoTransfer.py
```
