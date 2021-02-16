import os
import re
import face_recognition as fr
import dlib
import numpy as np
import random as r
import torch as t
from torchvision.transforms import Compose, Normalize, ToTensor, ToPILImage 
#from torch import autograd
from facenet_pytorch import InceptionResnetV1
from PIL import Image, ImageDraw, ImageChops

r.seed(1337)
device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

class Attack(object):
    def __init__(self, input_img, target_img, mask=None, optimizer='sgd', pretrained='vggface2'):
        self.normalize = Compose([
            ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        self.input = input_img
        self.target = target_img
        self.mask = None
        if (mask == None):
            self.mask = create_mask(self.input)
        else:
            self.mask = mask

        self.input_tensor = None
        self.input_emb = None
        self.target_emb = None
        self.loss = None

        self.resnet = InceptionResnetV1(pretrained=pretrained).eval().to(device)

        self.input_tensor = ToTensor()(self.input)
        self.input_emb = self.resnet(t.stack([self.normalize(self.input)]))
        self.target_emb = self.resnet(t.stack([self.normalize(self.target)]))

        try:
            if (optimizer == 'sgd'):
                self.opt = t.optim.SGD([self.mask], lr = 1e-1, momentum = 0.9, weight_decay = 0.0001)
            elif (optimizer == 'adam'):
                self.opt = t.optim.Adam([self.mask], lr = 1e-1, weight_decay = 0.0001)
            elif (optimizer == 'adamax'):
                self.opt = t.optim.Adamax([self.mask], lr = 1e-1, weight_decay = 0.0001)
        except:
            print("Optimizer not supported, reverting to ADAM")
            self.opt = t.optim.Adam([self.mask], lr = 1e-1, weight_decay = 0.0001)
    
    def train(self, epochs = 30):
        for i in range(epochs):
            adversarial_input = self._apply(self.input_tensor, self.mask)
            adversarial_emb = self.resnet(t.stack([adversarial_input]))

            self.loss = (-emb_distance(adversarial_emb, self.input_emb)
                        +emb_distance(adversarial_emb, self.target_emb))

            self.loss.backward(retain_graph=True)
            self.opt.step()
            self.opt.zero_grad()

            self.mask.data.clamp_(0,1)
            print(i,self.loss)
        return adversarial_input.detach()

    def _apply(self, image, mask):
        return t.where((mask == 0), image, mask)


def create_mask(face_image):
    """
    Helper function to create a facial mask to cover lower portion of the
    face.  Uses 'face_recognizer' library's landmark detector to build a
    list of tuples containing (x, y) coordinates of the lower chin area as 
    well as the middle of the nose tip.

    A polygon is then drawn using those tuples creating a "taco" shaped 
    face mask.  This is then processed for each channel with a value of 
    0 for white areas and a value of 1 for black areas (the taco area)

    This will later be used as a tensor that takes in these given values


    Parameters
    ----------
    face_image : PIL.Image
        image of a detected face

    Returns
    -------
    list : [np.array (float32), tuple]
        mask array and nose tip location
    """
    mask = Image.new('RGB', face_image.size, color=(255,255,255))
    d = ImageDraw.Draw(mask)
    
    landmarks = fr.face_landmarks(np.array(face_image))
    # Gross list comprehension magic
    area = [landmark 
            for landmark in landmarks[0]['chin']
            if landmark[1] > max(landmarks[0]['nose_tip'])[1]]
    area.append(landmarks[0]['nose_bridge'][1])
    
    d.polygon(area, fill=(0,0,0))
    mask_array = np.array(mask)
    mask_array = mask_array.astype(np.float32)
    # C
    for i in range(mask_array.shape[0]):
        # C
        for j in range(mask_array.shape[1]):
            # C
            for k in range(mask_array.shape[2]):
                # Combo BREAKER
                if mask_array[i][j][k] < 255.:
                    mask_array[i][j][k] = 0.5#r.random()
                else:
                    mask_array[i][j][k] = 0

    mask_tensor = ToTensor()(mask_array)
    mask_tensor.requires_grad_(True)

    return mask_tensor


def detect_face(image_file_loc):
    """
    Helper function to run the facial detection and alignment process using
    dlib.  Detects a given face and aligns it using dlib's 5 point landmark
    detector.


    Parameters
    ----------
    image_file_name : str
        image file location

    Returns
    -------
    list : PIL.Image
        Resized face image
    """
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor('./shape_predictor_5_face_landmarks.dat')
    image = dlib.load_rgb_image(image_file_loc)
    dets = detector(image, 1)

    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(shape_predictor(image, detection))

    face_image = Image.fromarray(dlib.get_face_chip(image, faces[0], size=300))

    return face_image


def emb_distance(tensor_1, tensor_2):
    tensor_1 = tensor_1
    tensor_2 = tensor_2
    ten = (tensor_1 - tensor_2)
    return ten.norm()


def load_data(path_to_data):
    """
    Helper function for loading image data.  Allows user to load the input, target, 
    and test images.  Mask creation and offsetting must be done manually.

    Parameters
    ----------
    path_to_data : str
        Path to the given data.  Ex: './faces/input/'

    Returns
    -------
    list : [[PIL.Image, tuple]]
        List of resized face images and nose tip locations
    """
    img_files = [f for f in os.listdir(path_to_data) if re.search(r'.*\.(jpe?g|png)', f)]
    img_files_locs = [os.path.join(path_to_data, f) for f in img_files]

    image_list = []

    for loc in img_files_locs:
        image_list.append(detect_face(loc))

    return image_list

# Testing 

inputs = load_data('./examples/faces/input/')

test = Attack(inputs[0], inputs[1])
x=test.train(5)





