# Introduction
### Learning Transferable Visual Models From Natural Language Supervision
![Clip](https://github.com/iljf/Assignment_CLIP/assets/94291960/78e0f6a9-0a76-4dd1-a3cf-71d86e73770a)
CLIP: Contrastive Language-Image Pre-training

https://arxiv.org/pdf/2103.00020.pdf

### Pseudocode
```
# image_encoder - ResNet or Vision Transformer
# text_encoder  - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l]       - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t             - learned temperature parameter

# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T)  #[n, d_t]

# joint multimodal embedding [n, d_e]
I_e = 12_normalize(np.dot(I_f, W_i), axis=1)
T_e = 12_normalize(np.dot(T_f, W_t), axis=1)

# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss   = (loss_i + loss_t) /2
```
- given a batch of N (image, text) pairs
- predict which of the N x N possible pairs across a batch actually occured
- CLIP learns the multi-modal embedding space by jointly training an image and text encoder to:
    - maximize the cosine similarity of the N real embedding pairs
    - minimize the cosine similarity of the N^2 - N incorrect embedding pairs
- optimize a symmetric cross entropy loss over the similarity scores
