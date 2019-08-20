# Face-Off
### Steps towards physical adversarial attacks on facial recognition

<img src="https://raw.githubusercontent.com/392781/Face-Off/master/results/example/input-face.png?token=AGA77BYF3VKF2GAE2TEGJR25JXOQU" width="175"> <img src="https://raw.githubusercontent.com/392781/Face-Off/master/results/example/delta.png?token=AGA77B2UH6QPPJJVJPV2IEK5JXOTY" width="175"> <img src="https://raw.githubusercontent.com/392781/Face-Off/master/results/example/combined-face.png?token=AGA77B3AA6CH3NHFOVQFUGS5JXOUS" width="175"> <img src="https://raw.githubusercontent.com/392781/Face-Off/master/results/example/target-face.png?token=AGA77B7ZOLHRLBGU5W7IMUC5JXOPC" width="175">

John Travolta is detected as Nicholas Cage after the mask has been applied


## Usage
Add `from adversarial_face_recognition import *` to be able to use the training classes + functions.

To get an idea on how to do experiments, check the 3 experiment files.

To get an idea on how the training procedure works without all the python class fluff obscuring the process check `example.py`

## References
* Sharif, Mahmood, et al. "Accessorize to a crime: Real and stealthy attacks on state-of-the-art face recognition." Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security. ACM, 2016.
* Wang, Mei, and Weihong Deng. "Deep face recognition: A survey." arXiv preprint arXiv:1804.06655 (2018).
* MacDonald, Bruce. “Fooling Facial Detection with Fashion.” Towards Data Science, Towards Data Science, 4 June 2019, towardsdatascience.com/fooling-facial-detection-with-fashion-d668ed919eb.
* Thys, Simen, et al. "Fooling automated surveillance cameras: adversarial patches to attack person detection." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2019.

Big thanks to Tim Esler for his [PyTorch FaceNet implementation](https://github.com/timesler/facenet-pytorch)
