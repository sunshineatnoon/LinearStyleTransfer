# Learning Linear Transformations for Fast Arbitrary Style Transfer

<img src="doc/images/chicago_paste.png" height="150" hspace="5"><img src="doc/images/photo_content.png" height="150" hspace="5"><img src="doc/images/content.gif" height="150" hspace="5">

<img src="doc/images/chicago_27.png" height="150" hspace="5"><img src="doc/images/in5_result.png" height="150" hspace="5"><img src="doc/images/test.gif" height="150" hspace="5">

## Prerequisites
- [Pytorch](http://pytorch.org/)
- [torchvision](https://github.com/pytorch/vision)
- [imageio](https://pypi.python.org/pypi/imageio) for GIF generation
- [opencv](https://opencv.org/) for video generation

## Image Style Transfer
- Clone from github: `git clone https://github.com/sunshineatnoon/LinearStyleTransfer`
- Download pre-trained models from [google drive](https://drive.google.com/open?id=1nBRM9KX4N0HV0uPItU1RuCved9q-vXdM)(131MB).
- Uncompress to root folder :
```
cd LinearStyleTransfer
tar -zxvf models.tar
```
- Artistic style transfer
```
python TestArtistic.py --vgg_dir models/vgg_normalised_conv4_1.t7 --decoder_dir models/feature_invertor_conv4_1.t7 --matrixPath models/layer4.pth --layer r41
```
or conduct style transfer on relu_31 features
```
python TestArtistic.py --vgg_dir models/vgg_normalised_conv3_1.t7 --decoder_dir models/feature_invertor_conv3_1.t7 --matrixPath models/layer3.pth --layer r31
```
- Photo-real style transfer
```
python TestPhotoReal.py --vgg_dir models/vgg_normalised_conv3_1.t7 --decoder_dir models/feature_invertor_conv3_1.t7 --matrixPath models/layer3.pth --layer r31
```
- Video style transfer
```
python TestVideo.py --vgg_dir models/vgg_normalised_conv3_1.t7 --decoder_dir models/feature_invertor_conv3_1.t7 --matrixPath models/layer3.pth --layer r31
```
- Real-time video demo
```
python real-time-demo.py --vgg_dir models/vgg_normalised_conv3_1.t7 --decoder_dir models/feature_invertor_conv3_1.t7 --matrixPath models/layer3.pth --layer r31
```

## Model Training
### Data Preparation
- MSCOCO
```
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
```
- WikiArt
  - Either manually download from [kaggle](https://www.kaggle.com/c/painter-by-numbers).
  - Or install [kaggle-cli](https://github.com/floydwch/kaggle-cli) and download by running:
  ```
  kg download -u <username> -p <password> -c painter-by-numbers -f train.zip
  ```

### Training
To train a model that transfers relu4_1 features, run:
```
python Train.py --vgg_dir models/vgg_normalised_conv4_1.t7 --decoder_dir models/feature_invertor_conv4_1.t7 --layer r41 --contentPath PATH_TO_MSCOCO --stylePath PATH_TO_WikiArt --outf OUTPUT_DIR
```
or train a model that transfers relu3_1 features:
```
python Train.py --vgg_dir models/vgg_normalised_conv3_1.t7 --decoder_dir models/feature_invertor_conv3_1.t7 --layer r31 --contentPath PATH_TO_MSCOCO --stylePath PATH_TO_WikiArt --outf OUTPUT_DIR
```
Key hyper-parameters:
- style_layers: which features to compute style loss.
- style_weight: larger style weight leads to heavier style in transferred images.

Intermediate results and weight will be stored in `OUTPUT_DIR`
