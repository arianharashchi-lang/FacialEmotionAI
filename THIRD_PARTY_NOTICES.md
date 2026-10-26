# Third-Party Notices

FacialEmotionAI is licensed under the MIT License (see `LICENSE`). The notices below apply to third-party browser libraries used by the optional phone-object detection feature; they do not replace or change the project's MIT license.

## TensorFlow.js

- Package: `@tensorflow/tfjs` (loaded from jsDelivr CDN)
- Version used by the page: `4.22.0`
- License: Apache License 2.0
- Upstream: https://github.com/tensorflow/tfjs

## TensorFlow.js COCO-SSD

- Package: `@tensorflow-models/coco-ssd` (loaded from jsDelivr CDN)
- Version used by the page: `2.2.3`
- License: Apache License 2.0
- Upstream package metadata: https://github.com/tensorflow/tfjs-models/blob/master/coco-ssd/package.json
- Model documentation and supported object classes: https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd

COCO-SSD can identify the COCO class `cell phone`. The optional detection runs in the browser using the user's camera stream; this project does not intentionally upload camera frames to its server for this feature.

The upstream projects and their license texts are authoritative. When redistributing copies of these libraries or their model artifacts, include the applicable Apache License 2.0 text and preserve required copyright, license, and NOTICE information from the distributed artifacts.
