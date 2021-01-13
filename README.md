# contextual_loss_tensorflow
I converted https://github.com/S-aiueo32/contextual_loss_pytorch to tensorflow version and fixed some bugs(l1, l2 dist func).

## Requirements
-  Python3.7+
-  `tensorflow` & `tensorflow.keras`

## Installation
```
pip install https://github.com/animilux/contextual_loss_tensorflow.git
```

## Usage
```python
import tensorflow as tf
import contextual_loss.fuctional as F

img1 = tf.random.uniform(shape=[1,32,32,3], minval=0., maxval=1.)
img2 = tf.random.uniform(shape=[1,32,32,3], minval=0., maxval=1.)

loss = F.contextual_loss(img1, img2, loss_type='l1', channel_last=True)

```

## Reference
### Papers
1. Mechrez, Roey, Itamar Talmi, and Lihi Zelnik-Manor. "The contextual loss for image transformation with non-aligned data." Proceedings of the European Conference on Computer Vision (ECCV). 2018.  
2. Mechrez, Roey, et al. "Maintaining natural image statistics with the contextual loss." Asian Conference on Computer Vision. Springer, Cham, 2018.
### Implementations
Thanks to the owners of the following awesome implementations.
- Original Repository: https://github.com/roimehrez/contextualLoss
- PyTorch Implemantation: https://github.com/S-aiueo32/contextual_loss_pytorch
