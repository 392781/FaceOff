import pickle
import glob
import PIL.Image
from adversarial_face_recognition import *

# Use GPU if available.
device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
apply = Applier()
apply = apply.to(device)

ground_truth_image = load_data('./faces/input/ronald.jpg')[0][0]
ground_truth_tensor = tensorize(ground_truth_image).to(device)

mask_loc = glob.glob('./results/exp_1/masks/*')
mask_loc.sort()
mask_list = []
for loc in mask_loc:
    with open(loc, 'rb') as f:
        mask_list.append(pickle.load(f))

# 0 - 39
id = 14
# 0 - 4
img = 3

adversarial = apply(ground_truth_tensor, mask_list[id][img]).detach()
masked_image = imagize(adversarial.cpu())

masked_image.show()