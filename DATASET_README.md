# CSCI 353 Project 1 Dataset

This is the dataset for Project 1.
The images are from the **manually aligned frontal images** from the FEI Face Database:

<https://fei.edu.br/~cet/facedatabase.html>

The images remain subject to FEI's terms. Do not upload the face images to an external AI service. 

## Dataset structure

The generator produces:

```text
dataset/
  aligned/
  easy/
  medium/
  large/
image_manifest.csv
recommended_evaluation_pairs.csv
```

Each selected portrait has four versions with the same `image_id`:

- `aligned`: an unshifted center crop;
- `easy`: a small translation (between 1 and 10 pixels);
- `medium`: a moderate translation (between 11 and 25 pixels); and
- `large`: a large translation (between 26 and 40 pixels).

Note that the indexes come in pairs.  For example, face_0001 and face_0002 are the same person with two different facial expressions.
The original FEI images are `360 x 260`. A 40-pixel maximum shift produces
`280 x 180` gallery images after cropping. 

You may use any pair of images for your hybrid images. 

## Manifests

`image_manifest.csv` contains one row per generated image. It records the
source identity and expression, gallery level, applied translation, inverse
alignment translation, crop dimensions, shift-band limits, and random seed.

For an individual image with applied translation `(dx, dy)`:

- positive `dx` moves its content right;
- negative `dx` moves its content left;
- positive `dy` moves its content down; and
- negative `dy` moves its content up.