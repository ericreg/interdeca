---
datasets:
- ILSVRC/imagenet-1k
library_name: transformers
license: cc-by-nc-4.0
---

# I-JEPA Model (Huge, fine-tuned on IN1K)

**I-JEPA** is a method for self-supervised learning. At a high level, I-JEPA predicts the representations of part of an image from the representations of other parts of the same image:
1. without relying on pre-specified invariances to hand-crafted data transformations, which tend to be biased for particular downstream tasks,
2. and without having the model fill in pixel-level details, which tend to result in learning less semantically meaningful representations.

![ijepa](https://github.com/facebookresearch/ijepa/assets/7530871/dbad94ab-ac35-433b-8b4c-ca227886d311)


## How does it work?

As opposed to generative methods that have a pixel decoder, I-JEPA has a predictor that makes predictions in latent space.
The predictor in I-JEPA can be seen as a primitive (and restricted) world-model that is able to model spatial uncertainty in a static image from a partially observable context.
This world model is semantic in the sense that it predicts high level information about unseen regions in the image, rather than pixel-level details.

We trained a stochastic decoder that maps the I-JEPA predicted representations back in pixel space as sketches.
The model correctly captures positional uncertainty and produces high-level object parts with the correct pose (e.g., dog’s head, wolf’s front legs).

![Illustrating how the predictor learns to model the semantics of the world](https://github.com/facebookresearch/ijepa/assets/7530871/9b66e461-fc8b-4b12-9f06-63ec4dfc1452)

## Intended uses & limitations

I-JEPA can be used for image classification or feature extraction. This checkpoint in specific is intended for **Feature Extraction**.

## How to use

Here is how to use this model for image feature extraction:

```python
import requests
from PIL import Image
from torch.nn.functional import cosine_similarity

from transformers import AutoModel, AutoProcessor

url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
url_2 = "http://images.cocodataset.org/val2017/000000219578.jpg"
image_1 = Image.open(requests.get(url_1, stream=True).raw)
image_2 = Image.open(requests.get(url_2, stream=True).raw)

model_id = "facebook/ijepa_vith14_1k"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)


def infer(image):
    inputs = processor(image, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


embed_1 = infer(image_1)
embed_2 = infer(image_2)

similarity = cosine_similarity(embed_1, embed_2)
print(similarity)
```

### BibTeX entry and citation info
If you use I-JEPA or this code in your work, please cite:
```
@article{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2301.08243},
  year={2023}
}
```