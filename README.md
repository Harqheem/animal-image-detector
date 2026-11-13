# Fieldmark

Fieldmark is an animal detector. A convolutional network written in this repo looks at a photograph and draws a box around each bird, cat, dog, horse, sheep, or cow it is willing to name. The page runs that network in the browser. The photograph is not uploaded.

## Run the detector

```bash
npm install
npm run dev
```

Open the local URL printed by Next.js. Drop in a photograph, or start from one of the sample frames.

## The network

`ml/model.py` defines `AnimalDetector`. A 224×224 image passes through a ResNet-18 trunk that starts from ImageNet weights, so the last map is a 7×7 grid of 512 channels. The detection head on that grid is trained in this project. Each cell has two predictors. A predictor outputs:

- the box center inside that cell
- the box width and height, as fractions of the image
- an objectness logit
- one logit per animal class

Training uses the loss in `ml/loss.py` on animal boxes from Pascal VOC 2007. People, vehicles, and furniture in those photos are background. The trunk stays frozen for the first epochs while the head learns, then the whole network is fine-tuned. An extra class head on the pooled features is used only while training and is not part of the exported network. The images themselves are not part of this repo. Download the trainval archive and extract it so `data/VOCdevkit/VOC2007` exists, then:

```bash
pip install -r requirements.txt
python -m ml.self_check
python -m ml.train
```

`python -m ml.self_check` memorizes four synthetic rectangles. If that recall is short of perfect, the grid math is wrong.

`python -m ml.train` writes:

- `public/models/animal-detector.onnx` — the weights the page runs
- `public/models/animal-detector.pt` — the same weights for `python -m ml.predict photo.jpg`
- `public/models/spec.json` — grid size and class names
- `public/models/report.json` — held-out recall and the layer list

The network only knows those six animals. A lion, a spider, or a drawing can stay unmarked, and a held-out recall well under a perfect score is expected for a detector trained from scratch on this set.
