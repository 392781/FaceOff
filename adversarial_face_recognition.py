import torch.optim as optim
import torch as t
import torch.nn as nn
import torchvision.transforms as transforms
import dlib
import numpy as np
import random as r
import face_recognition as fr
from tqdm import tqdm
from torch import autograd
from PIL import Image, ImageDraw, ImageChops
from facenet_pytorch import MTCNN, InceptionResnetV1

## Initalizes a constant random seed to keep results consistent if random
## mask is being used 
r.seed(1)
tensorize = transforms.ToTensor()
imagize = transforms.ToPILImage()

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
    """
    Applies the tensor mask onto the given image
    """
    def __init__(self):
        super(Applier, self).__init__()

    def forward(self, image, mask):
        """
        Applies a mask on an image


        Parameters
        ----------
        image : Tensor
            face tensor

        mask : Tensor
            calculated mask tensor

        Returns
        -------
        Tensor
            combined image and mask tensor
        """
        adversarial_tensor = t.where((mask == 0), image, mask)
        return adversarial_tensor



class Attack(object):
    """
    Class used to create adversarial facial recognition attacks
    """
    def __init__(self, input_list, target_list, mask_list, optimizer):
        """
        Class initialization with lists of preprocessed inputs, targets, and masks
        There are 3 optimizer options: SGD, Adam, Adamax


        Parameters
        ----------
        input_list : list[PIL.Image]
            list of inputs to train on

        target_list : list[PIL.Image]
            list of targets

        mask_list : list[np.array]
            list of preprocessed masks to attach to the input

        optimizer : str
            takes in either 'sgd', 'adam', or 'adamax'
        """
        self.input_list = input_list
        self.target_list = target_list
        self.mask_list = mask_list

        self.input_tensors = []
        self.input_emb = []
        self.target_emb = []
        self.losses = []

        # Necessary tools for training: normalization, image + delta applier, and 
        # facial recognition model
        self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.apply = Applier()
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

        # Read all inputs in.  Embeddings will be used for loss calculation, tensors
        # will be used for actual training
        for image, _ in self.input_list:
            self.input_emb.append(self.resnet(self.norm(tensorize(image))))
            self.input_tensors.append(tensorize(image))

        # Create target embeddings for loss calculation
        for image, _ in self.target_list:
            self.target_emb.append(self.resnet(self.norm(tensorize(image))))

        try:
            if (optimizer is 'sgd'):
                self.opt = optim.SGD(self.mask_list, lr = 1e-1, momentum = 0.9, weight_decay = 0.0001)
            elif (optimizer is 'adam'):
                self.opt = optim.Adam(self.mask_list, lr = 1e-1, weight_decay = 0.0001)
            elif (optimizer is 'adamax'):
                self.opt = optim.Adamax(self.mask_list, lr = 1e-1, weight_decay = 0.0001)
        except:
            print("No optimizer chosen, reverting to ADAM")
            self.opt = optim.Adam(self.mask_list, lr = 1e-1, weight_decay = 0.0001)
    
    def train(self, epochs = 30):
        """
        Trainer.  Essentially the process found in example.py but blown up to work 
        with lists of objects for batch training.

        Parameters
        ----------
        epochs : int
            number of rounds to train
        """

        # Initialize lists for each individual input to train and calculate loss on
        self.adversarial_list = [None for i in range(len(self.input_tensors))]
        embeddings = [None for i in range(len(self.input_tensors))]
        self.losses = [None for i in range(len(self.input_tensors))]

        # Begin training
        print('\nTRAINING -------')
        for i in tqdm(range(epochs)):
            # For each image, run this training process:
            for i in range(len(self.adversarial_list)):
                # Applies the mask onto the image
                self.adversarial_list[i] = apply(self.input_tensors[i], self.mask_list[i])
                # Calculates the embedding of this adversarial image
                embeddings[i] = self.resnet(self.norm(self.adversarial_list[i]))
                # Calculates loss: Maximizes distance between adversarial image and 
                # input image while minimizing the distance between adversarial image
                # and target image
                self.losses[i] = (-emb_distance(embeddings[i], self.input_emb[i])
                                  +emb_distance(embeddings[i], self.target_emb[i]))
                
                # Performs backprop based on loss
                self.losses[i].backward(retain_graph=True)
                self.opt.step()
                self.opt.zero_grad()

                # ... which updates the mask ... and the process begins again
                self.mask_list[i].data.clamp_(-1, 1)
        print(self.losses)

    def results(self, input_test_list, target_test_list, save_path='/'):
        """
        Displays results based on two test image lists: one for input, one for target


        Parameters
        ----------
        input_test_list : list[PIL.Image]
            test images for the inputs

        target_test_list : list[PIL.Image]
            test images for the targets

        save_path : str
            specify a save path further insice './results/{save_path}'
            default just keeps './results/[delta, combined, ...]/' file structure
        """

        # Just a LOT of euclidean distance calculations
        print('\ninputs vs ground truths')
        for i in range(len(self.input_list)):
            input_test_emb = self.resnet(self.norm(tensorize(input_test_list[i])))
            print(emb_distance(self.input_emb[i], input_test_emb).item())

        print('\ninputs vs target')
        for i in range(len(self.input_list)):
            print(emb_distance(self.input_emb[i], self.target_emb[i]).item())

        print('\ninputs vs target 2')
        for i in range(len(self.input_list)):
            target_test_emb = self.resnet(self.norm(tensorize(target_test_list[i])))
            print(emb_distance(self.input_emb[i], target_test_emb).item())

        print('\ntarget vs target 2')
        for i in range(len(self.input_list)):
            target_test_emb = self.resnet(self.norm(tensorize(target_test_list[i])))
            print(emb_distance(self.target_emb[i], target_test_emb).item())

        print('\nadversarial vs ground truths')
        for i in range(len(self.input_list)):
            adversarial_emb = self.resnet(self.norm(self.apply(self.input_tensors[i],
                                                self.mask_list[i])))
            input_test_emb = self.resnet(self.norm(tensorize(input_test_list[i])))                                    
            print(emb_distance(adversarial_emb, input_test_emb).item())
        
        print('\nadversarial vs target')
        for i in range(len(self.input_list)):
            adversarial_emb = self.resnet(self.norm(self.apply(self.input_tensors[i],
                                                self.mask_list[i])))
            print(emb_distance(adversarial_emb, self.target_emb[i]).item())

        print('\nadversarial vs target 2')
        for i in range(len(self.input_list)):
            adversarial_emb = self.resnet(self.norm(self.apply(self.input_tensors[i],
                                                self.mask_list[i])))
            target_test_emb = self.resnet(self.norm(tensorize(target_test_list[i])))
            print(emb_distance(adversarial_emb, target_test_emb).item())

        print('\nSAVING IMAGES')
        for i in tqdm(range(len(self.input_list))):
            imagize(self.mask_list[i].detach()).save(f'./results{save_path}delta/{i}.png')
            imagize((self.input_tensors[i] + self.mask_list[i]).detach()).save(
                f'./results{save_path}combined/{i}.png'
            )
            self.input_list[i][0].save(f'./results{save_path}input/{i}.png')
            self.target_list[i][0].save(f'./results{save_path}target/{i}.png')


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



def mask_offset(image, mask, mask_coor):
    """
    Offsets the mask based on the nose location of the created mask and the target 
    image


    Parameters
    ----------
    image : list[PIL.Image, tuple]
        target image to align mask to

    mask : np.array
        mask to align

    mask_coor : tuple
        mask nose tip coordinate

    Returns
    -------
    PIL.Image
        Image of the offset mask
    """
    dist = (image[1][0] - mask_coor[0], image[1][1] - mask_coor[1])
    new_mask = ImageChops.offset(imagize(tensorize(mask)), dist[0], dist[1])
    new_mask = tensorize(new_mask)
    new_mask.requires_grad_(True)
    return new_mask



def detect_face(image_file_name):
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
    list : [PIL.Image, tuple]
        Resized face image and nose tip location
    """
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor('./shape_predictor_5_face_landmarks.dat')
    image = dlib.load_rgb_image(image_file_name)
    dets = detector(image, 1)

    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(shape_predictor(image, detection))

    face_image = Image.fromarray(dlib.get_face_chip(image, faces[0], size=300))
    landmarks = fr.face_landmarks(np.array(face_image))
    

    return [face_image, landmarks[0]['nose_tip'][2]]

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

    return [mask_array, landmarks[0]['nose_tip'][2]]
