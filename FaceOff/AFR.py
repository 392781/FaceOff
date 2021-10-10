import os
import re
import dlib
import face_recognition as fr
import face_recognition_models as frm
import numpy as np
import torch as t
from torchvision.transforms import Compose, Normalize, ToTensor, ToPILImage 
from facenet_pytorch import InceptionResnetV1
from PIL import Image, ImageDraw, ImageChops

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class Attack(object):
    """
    Class used to create adversarial facial recognition attacks
    """
    def __init__(self, 
        input_img, 
        target_img, 
        seed=None, 
        optimizer='sgd', 
        lr = 1e-1, 
        pretrained='vggface2'
    ):
        """
        Initialization for Attack class.  Attack contains the following:
            input_tensor
            input_emb
            target_emb
            mask_tensor
            ref (mask reference for _apply)

        Parameters
        ----------
        input_img : PIL.Image
            Image to train on.

        target_img : PIL.Image
            Image to target the adversarial attack against.

        seed : int, optional
            Sets custom seed for reproducability. Default is generated randomly.

        optimizer : str, optional
            Takes in either 'sgd', 'adam', or 'adamax'.  Default is 'adam'.

        lr : float, optional
            Learning rate.  Default is 1e-1 or 0.1.
        
        pretrained : str, optional
            Pretrained weights for FaceNet.  Options are 'vggface2' or 'casia-webface'.
            Default is 'vggface2'.
        """
        # Value inits
        if (seed != None) : np.random.seed(seed)
        self.MEAN = t.tensor([0.485, 0.456, 0.406], device=device)
        self.STD = t.tensor([0.229, 0.224, 0.225], device=device)
        self.LOSS = t.tensor(0, device=device)

        # Function inits
        self.imageize = ToPILImage()
        self.tensorize = ToTensor()
        self.normalize = Normalize(mean=self.MEAN.cpu().numpy(), std=self.STD.cpu().numpy())
        self.resnet = InceptionResnetV1(pretrained=pretrained).eval().to(device)
        
        # Training inputs
        # Original image - normalized and with embedding created.
        self.input_tensor = self.normalize(self.tensorize(input_img).to(device))
        self.input_emb = self.resnet(
            t.stack([
                    self.input_tensor
            ])
        )
        # Target image - normalized and with embedding created.
        self.target_emb = self.resnet(
            t.stack([
                self.normalize(
                    self.tensorize(target_img).to(device)
                )
            ])
        )
        # Adversarial embedding init
        self.adversarial_emb = None
        # Face mask init
        self.mask_tensor = self._create_mask(input_img)
        # Reference tensor used to apply mask
        self.ref = self.mask_tensor

        # Optimizer init
        try:
            if (optimizer == 'sgd'):
                self.opt = t.optim.SGD([self.mask_tensor], lr = lr, momentum = 0.9, weight_decay = 0.0001)
            elif (optimizer == 'adam'):
                self.opt = t.optim.Adam([self.mask_tensor], lr = lr, weight_decay = 0.0001)
            elif (optimizer == 'adamax'):
                self.opt = t.optim.Adamax([self.mask_tensor], lr = lr, weight_decay = 0.0001)
        except:
            print("Optimizer not supported, reverting to ADAM")
            self.opt = t.optim.Adam([self.mask_tensor], lr = lr, weight_decay = 0.0001)
    

    def train(self, 
        epochs = 30, 
        detect=False, 
        verbose=False
    ):
        """
        Adversarial training process for facial recognition.

        Parameters
        ----------
        epochs : int, optional
            Number of training epochs.  Default is 30.

        detect : bool, optional
            Perform facial detection during training process and log result.  Default is False.

        verbose : bool, optional
            Output full embedding distance information during training.  Default is False.

        Returns
        -------
        list
            Adversarial tensor, mask tensor, adversarial image
        """
        for i in range(1, epochs + 1):
            # Create adversarial tensor by applying normalized MASK to normalized INPUT
            self.view(self.mask_tensor).show()
            adversarial_tensor = self._apply(
                self.input_tensor, 
                self.normalize(self.mask_tensor),
                self.ref)
            # Create embedding
            self.adversarial_emb = self.resnet(t.stack([adversarial_tensor]))

            # Calculate two distances - from adv to input and adv to target
            distance_to_image = self._emb_distance(self.adversarial_emb, self.input_emb)
            distance_to_target = self._emb_distance(self.adversarial_emb, self.target_emb)
            # Calculate loss - maximize distance to image, minimize distance to target
            self.LOSS = (-distance_to_image + distance_to_target)
            # Adversarially backpropagate the information back to original adversarial_tensor
            self.LOSS.backward(retain_graph=True)
            # Update optimizer and zero out gradients 
            self.opt.step()
            self.opt.zero_grad()
            # Allow the updated mask to have only values (0, 255) or normalized (0, 1)
            self.mask_tensor.data.clamp_(0, 1)

            # Various logging information
            training_information = [f'Epoch {i}: \n   Loss            = {self.LOSS.item():.7f}']
            if verbose:
                training_information.append(f'\n   Dist. to Image  = {distance_to_image:.7f}')
                training_information.append(f'\n   Dist. to Target = {distance_to_target:.7f}')
            if detect:
                face_loc = fr.face_locations(np.array(self.imageize(self._reverse_norm(adversarial_tensor).detach())))
                detected = False if not face_loc else True
                training_information.append(f'\n   Face detection  = {detected}')
            print(''.join(training_information))    

        # Return original adversarial tensor, the adversarial image, and the mask tensor
        adversarial_image = self.imageize(self._reverse_norm(adversarial_tensor).detach())
        return adversarial_tensor, self.mask_tensor, adversarial_image


    def view(self, 
        norm_image_tensor, 
        norm_mask_tensor=None
    ):
        """
        Preview a tensor as an image

        Parameters
        ----------
        norm_image_tensor : torch.Tensor
            Image to convert.
        norm_mask_tensor : torch.Tensor, optional
            Mask to apply to image. Default is None.
        
        Returns
        -------
        PIL.Image
        """
        if norm_mask_tensor is not None:
            combined_tensor = self._apply(norm_image_tensor, norm_mask_tensor, self.ref)
            return self.imageize(self._reverse_norm(combined_tensor).detach())
        else:
            return self.imageize(self._reverse_norm(norm_image_tensor).detach())


    def _apply(self, 
        image_tensor, 
        mask_tensor, 
        reference_tensor
    ):
        """
        Apply a mask over an image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Canvas to be used to apply mask on.

        mask_tensor : torch.Tensor
            Mask to apply over the image.

        reference_tensor : torch.Tensor
            Used to reference mask boundaries

        Returns
        -------
        torch.Tensor
        """
        return t.where((reference_tensor == 0), image_tensor, mask_tensor).to(device)


    def _create_mask(self, face_image):
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
        mask_tensor : torch.Tensor
            mask tensor
        """
        # New image mask
        mask = Image.new('RGB', face_image.size, color=(0,0,0))
        # Draw on mask
        d = ImageDraw.Draw(mask)
        
        # Detect face landmarks
        landmarks = fr.face_landmarks(np.array(face_image))
        # Gross list comprehension magic to add coordinates from image to create face mask
        area = [landmark 
                for landmark in landmarks[0]['chin']
                if landmark[1] > max(landmarks[0]['nose_tip'])[1]]
        area.append(landmarks[0]['nose_bridge'][1])
        
        # Fill the new face mask area
        d.polygon(area, fill=(255,255,255))
        # Create a numpy array and set type as floating point (used in training)
        mask_array = np.array(mask)
        mask_array = mask_array.astype(np.float32)

        # Fill the mask in with the color grey
        # C
        for i in range(mask_array.shape[0]):
            # C
            for j in range(mask_array.shape[1]):
                # C
                for k in range(mask_array.shape[2]):
                    # Combo BREAKER
                    if mask_array[i][j][k] == 255.:
                        mask_array[i][j][k] = 0.5
                    else:
                        mask_array[i][j][k] = 0

        # Create the mask tensor and initialize gradient
        mask_tensor = ToTensor()(mask_array).to(device)
        mask_tensor.requires_grad_(True)

        return mask_tensor


    def _emb_distance(self, tensor_1, tensor_2):
        """
        Helper function to calculate Euclidean distance between two tensors.

        Parameters
        ----------
        tensor_1, tensor_2 : torch.Tensor
            Tensors used for distance calculation

        Returns
        distance_tensor : torch.Tensor
            Tensor containing distance value
        -------

        """
        distance_tensor = (tensor_1 - tensor_2).norm()
        return distance_tensor


    def _reverse_norm(self, image_tensor):
        """
        Reverses normalization for a given image_tensor

        Parameters
        ----------
        image_tensor : torch.Tensor
        
        Returns
        -------
        torch.Tensor
        """
        return image_tensor * self.STD[:, None, None] + self.MEAN[:, None, None]



def detect_face(image_loc):
    """
    Helper function to run the facial detection and alignment process using
    dlib.  Detects a given face and aligns it using dlib's 5 point landmark
    detector.

    Parameters
    ----------
    image_loc : numpy.array
        image file location

    Returns
    -------
    face_image : PIL.Image
        Resized face image
    """
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(frm.pose_predictor_model_location())
    image = dlib.load_rgb_image(image_loc)
    dets = detector(image, 1)

    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(shape_predictor(image, detection))

    face_image = Image.fromarray(dlib.get_face_chip(image, faces[0], size=300))

    return face_image


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
    list : [PIL.Image]
        List of resized face images
    """
    img_files = [f for f in os.listdir(path_to_data) if re.search(r'.*\.(jpe?g|png)', f)]
    img_files_locs = [os.path.join(path_to_data, f) for f in img_files]

    image_list = []

    for loc in img_files_locs:
        image_list.append(detect_face(loc))

    return image_list
