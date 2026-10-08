# Human Action Recognition with YOLO11-Pose + LSTM

Recognize human actions (jumping rope, push-ups, squats, ...) from video using body keypoints.
A pretrained **YOLO11-Pose** model extracts the skeleton of the person in every frame, and a lightweight **LSTM** classifies the motion sequence. The output is the original video with the skeleton drawn on it and the predicted action shown in a label box.

## Overview

Instead of training a heavy model on raw video, this project works on **pose keypoints**:

```
Video -> YOLO11-Pose -> 17 keypoints/frame -> 30-frame sequence -> LSTM -> Action label
```

This makes training fast, light on memory, and independent of background and clothing.

## Features

- Skeleton extraction with a pretrained `yolo11n-pose` model (no pose training needed)
- Automatic selection of the main person when several people are detected
- Fixed-length sequences (30 evenly sampled frames per video)
- Normalization that removes position and scale (centered on the hips, scaled by torso length)
- LSTM classifier built with TensorFlow/Keras
- Annotated output video with skeleton and a class label box

## Dataset

[UCF101](https://www.crcv.ucf.edu/data/UCF101.php), Kaggle version: *UCF101 - Action Recognition* (Matthew Jansen). Uses the provided `train` and `test` folders.

The model is trained on **10 classes** with clear body motion:

| | | | | |
|---|---|---|---|---|
| JumpRope | JumpingJack | PushUps | PullUps | BodyWeightSquats |
| Lunges | Punch | TaiChi | HandstandPushups | SumoWrestling |

## Pipeline

1. **Keypoint extraction**: YOLO11-Pose returns 17 COCO keypoints (x, y) per frame.
2. **Person selection**: if several people are found, the one with the largest bounding box is used; if nobody is found, the frame is filled with zeros.
3. **Length fixing**: 30 frames are sampled uniformly from each video.
4. **Normalization**: keypoints are centered on the hip midpoint and divided by the shoulder-to-hip distance.
5. **Flattening**: each sample becomes a `(30, 34)` array (17 points x 2 coordinates).
6. **Classification**: a two-layer LSTM with dropout and a softmax output.

## Model

```
Input (30, 34)
 -> LSTM(128, return_sequences=True) -> Dropout(0.3)
 -> LSTM(64) -> Dropout(0.3)
 -> Dense(num_classes, softmax)
```

Optimizer: Adam. Loss: sparse categorical cross-entropy. Epochs: 40. Batch size: 16.

## Results

| Setup | Test accuracy |
|---|---|
| 5 classes, 30 epochs | ~87% |
| 10 classes, 40 epochs | %90 |

> Note: the confidence value drawn on the video is the softmax output, not the real accuracy. Use `model.evaluate` on the test set for the true accuracy.

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install ultralytics tensorflow opencv-python scikit-learn numpy tqdm pillow
```

`ffmpeg` is needed to convert output videos to a browser-friendly `.mp4`.

## Usage

**1. Extract keypoints and save them**

```python
X, y = [], []
for label, cls in enumerate(classes):
    for name in os.listdir(os.path.join(root, cls)):
        X.append(video_to_seq(os.path.join(root, cls, name)))
        y.append(label)

np.save("X_train.npy", np.array(X))
np.save("y_train.npy", np.array(y))
```

**2. Train**

```python
model_lstm.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=40, batch_size=16)
model_lstm.save("pose_lstm.keras")
```

**3. Evaluate**

```python
model_lstm.evaluate(X_test, y_test)
```

**4. Predict and draw on a video**

```python
draw_video("path/to/video.avi", out_path="result.mp4")
```


## Limitations

- Only the 10 trained classes are recognized; any other action will be forced into one of them.
- One prediction is made per whole video, not per time window.
- Similar movements (`PushUps`, `HandstandPushups`, `PullUps`) can be confused.
- Actions that depend on objects or scenes are hard to separate using the skeleton alone.

## Future Work

- Add velocity features (frame-to-frame keypoint differences)
- Data augmentation (rotation, scaling, horizontal flip)
- Sliding-window, real-time prediction
- Bidirectional LSTM or Transformer
- Scale to all 101 UCF101 classes

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php)
