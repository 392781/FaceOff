from FaceOff.AFR import load_data, Attack
from PIL import Image

# Load the data.  This will detect and resize the faces
inputs = load_data('./faces/input/')
targets = load_data('./faces/target/')

# Initialize the Attack object with 
adversarial = Attack(inputs[0], targets[3], optimizer='adam')

# Perform adversarial training
adversarial_tensor, mask_tensor, img = adversarial.train(detect=True, verbose=True)

# Show the image with mask applied
img.show()