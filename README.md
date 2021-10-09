# FaceOff
### Steps towards physical adversarial attacks on facial recognition

<img src="https://raw.githubusercontent.com/392781/FaceOff/master/examples/faces/input-face-example.png" width="175"> <img src="https://raw.githubusercontent.com/392781/FaceOff/master/examples/faces/delta-example.png" width="175"> <img src="https://raw.githubusercontent.com/392781/FaceOff/master/examples/faces/combined-face-example.png" width="175"> <img src="https://raw.githubusercontent.com/392781/FaceOff/master/examples/faces/target-face-example.png" width="175">

Input image on the left is detected as the target image on the right after the mask has been applied.


## Installation
1. Create a virtual environment

```bash
conda create -n facial pip
```

2. Clone the repo 

```git
git clone https://github.com/392781/FaceOff.git
```

3. Install the required libraries 

```bash
pip install -r requirements.txt
```

4. Import and use!

```python
from adversarial_face_recognition import *`
```

For training instructions look at [`example.py`](https://github.com/392781/FaceOff/blob/master/examples/example.py) to get started in less than 30 lines.

## Usage
The purpose of this library is to create adversarial attacks agains the FaceNet face recognizer.  This is the preliminary work towards creating a more robust physical attack using a mask that a person could wear over their face.

The current pipeline consists of an aligned input image with a calculated mask.  This is then fed into a face detector using dlib's histogram of oriented gradients detector to test whether the face is still detected.  This is then passed to FaceNet where which ouputs a face embedding and a loss which is then calculated and propogated back.  This perturbs the input mask which generates enough of a disturbance to affect the loss.

The loss function maximizes the Euclidean distance between the inputs' true identity and minimizes the distance between the adversarial input and the target image.

An image of this process can be seen below.

<img src="https://raw.githubusercontent.com/392781/FaceOff/master/procedure.png">

## Citation
Please cite `FaceOff` if used in your research:

```tex
@misc{FaceOff,
  author = {Ronald Lencevičius},
  howpublished = {GitHub},
  title = {Face-Off: Steps towards physical adversarial attacks on facial recognition},
  URL = {https://github.com/392781/FaceOff},
  year = {2019},
}
```

## References
* Sharif, Mahmood, et al. "Accessorize to a crime: Real and stealthy attacks on state-of-the-art face recognition." Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security. ACM, 2016.
* Wang, Mei, and Weihong Deng. "Deep face recognition: A survey." arXiv preprint arXiv:1804.06655 (2018).
* MacDonald, Bruce. “Fooling Facial Detection with Fashion.” Towards Data Science, Towards Data Science, 4 June 2019, towardsdatascience.com/fooling-facial-detection-with-fashion-d668ed919eb.
* Thys, Simen, et al. "Fooling automated surveillance cameras: adversarial patches to attack person detection." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2019.

Used the [PyTorch FaceNet implementation](https://github.com/timesler/facenet-pytorch) by Tim Esler
