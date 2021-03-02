# Bird box

Software for a device designed to detect and classify birds using computer vision.

This repository is for showcasing only and not complete.

# Example results

![](example1.gif)
![](example2.gif)

# Overview
- I use a pretrained model from the [tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) to detect birds.
- Object tracking is done with the [pysot](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) implementation of the [SiamRPN](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html) tracker.
- The birds are classified with a self-trained ["EfficientNet"](https://arxiv.org/pdf/1905.11946.pdf) CNN.

Thanks to [Andreas Koch](https://koch-visuals.com/) for providing the bird footage.