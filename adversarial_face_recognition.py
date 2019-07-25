import torch.optim as optim
import torch as t
import torch.nn as nn
import torchvision.transforms as transforms
import dlib
import numpy as np
import random as r
import face_recognition as fr
from torch import autograd
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, InceptionResnetV1

r.seed(1)

class Normalize(nn.Module):
    """
    Class to normalize a given image
    """
    def __init__(self, mean, std):
        """
        Parameters
        ----------
        mean : list
            a list of mean values for each given dimension
        std : list
            a list of standard deviation values fro each given dimension
        """
        super(Normalize, self).__init__()
        self.mean = t.Tensor(mean)
        self.std = t.Tensor(std)
    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            a PyTorch Tensor of an image

        Returns
        -------
        Tensor
            a normalized image tensor
        """
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]



class Applier(nn.Module):
    def __init__(self):
        super(Applier, self).__init__()
    def forward(self, img, adv):
        img = t.where((adv == 0), img, adv)
        return img



def emb_distance(tensor_1, tensor_2):
    """
    Finds the Euclidean distance between two given image tensors


    Parameters
    ----------
    tensor_1 : Tensor
        image tensor to compare
    tensor_2 : Tensor
        image tensor to compare

    Returns
    -------
    Tensor
        single item tensor containing the distance between tensors
        access by calling item()
    """
    return (tensor_1 - tensor_2).norm()



def detect_face(image_file_name):
    """
    Helper function to run the facial detection process.  Uses the 
    'face_recognizer' library (which runs dlib's face detector).  This finds
    the top, right, bottom, left face borders and selects just the given 
    face, resizes, and resamples the image.


    Parameters
    ----------
    image_file_name : str
        image file location

    Returns
    -------
    PIL.Image
        Resized face image
    """
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor('./tools/shape_predictor_5_face_landmarks.dat')
    image = dlib.load_rgb_image(image_file_name)
    dets = detector(image, 1)

    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(shape_predictor(image, detection))

    return Image.fromarray(dlib.get_face_chip(image, faces[0], size=300))


"""
    image = fr.load_image_file(image_file_name)
    face_locations = fr.face_locations(image)
    top, right, bottom, left = face_locations[0]
    face_array = image[top:bottom, left:right]
    face_image = Image.fromarray(face_array)
    face_image = face_image.resize(size=(300,300), resample=Image.LANCZOS)
    return face_image
"""


def create_mask(face_image, mask_type = 'white'):
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
    np.array (float32)
        mask array
    """
    mask = Image.new('RGB', face_image.size, color=(255,255,255))
    d = ImageDraw.Draw(mask)
    
    landmarks = fr.face_landmarks(np.array(face_image))
    # Gross list comprehension magic
    area = [landmark 
            for landmark in landmarks[0]['chin']
            if landmark[1] > max(landmarks[0]['nose_tip'])[1]]
    area.append(landmarks[0]['nose_tip'][2])
    
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

    return mask_array


############################ MAIN ############################


norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
apply = Applier()
tensorize = transforms.ToTensor()
imagize = transforms.ToPILImage()
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

input_image_location =  './faces/ronald.jpg'
target_image_location = './faces/john.jpg'
input_test_location =   './faces/ronald2.jpg'
target_test_location =  './faces/john2.jpg'

input_image = detect_face(input_image_location)
target_image = detect_face(target_image_location)
input_image.save('./results/input-face.png')
target_image.save('./results/target-face.png')

mask = create_mask(input_image)
delta = tensorize(mask)
delta.requires_grad_(True)

opt = optim.Adam([delta], lr = 1e-1, weight_decay = 0.0001)

input_emb = resnet(norm(tensorize(input_image)))
target_emb = resnet(norm(tensorize(target_image)))

input_tensor = tensorize(input_image)

epochs = 45

print(f'\nEpoch |   Loss   | Face Detection')
print(f'---------------------------------')
for i in range(epochs):
    adver = apply(input_tensor, delta)
    adv = imagize(adver.detach())
    embedding = resnet(norm(adver))
    loss = (-emb_distance(embedding, input_emb)
            +emb_distance(embedding, target_emb))

    if i % 5 == 0 or i == epochs - 1:
        detection_test = fr.face_locations(np.array(adv))
        if not detection_test:
            d = 'Failed'
        else:
            d = 'Pass ' + str(detection_test)
        print(f'{i:5} | {loss.item():8.5f} | {d}')
        
        adv.show()

    loss.backward(retain_graph=True)
    opt.step()
    opt.zero_grad()

    delta.data.clamp_(-1, 1)

temp = detect_face(input_test_location)
true_emb = resnet(norm(tensorize(temp)))
temp = detect_face(target_test_location)
test_emb = resnet(norm(tensorize(temp)))

print("\ninput img vs true img  ", emb_distance(input_emb, true_emb).item())
print("input img vs target    ", emb_distance(input_emb, target_emb).item())
print("input img vs 2nd target", emb_distance(input_emb, test_emb).item())
print(" target vs 2nd target  ", emb_distance(target_emb, test_emb).item())
print("advr img vs true img   ", emb_distance(resnet(norm(apply(input_tensor, delta))), true_emb).item())
print("advr img vs target     ", emb_distance(resnet(norm(apply(input_tensor, delta))), target_emb).item())
print("advr img vs 2nd target ", emb_distance(resnet(norm(apply(input_tensor, delta))), test_emb).item())

imagize(delta.detach()).show()
imagize(delta.detach()).save('./results/delta.png')
imagize((input_tensor + delta).detach()).save('./results/combined-face.png')
