import pickle
import glob
from tqdm import tqdm, trange
from adversarial_face_recognition import *

# Use GPU if available.
device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

with open('./database.file', 'rb') as f:
    database = pickle.load(f)

# Image normalization using ImageNet normalization values
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
norm = norm.to(device)

# Used to apply a mask onto an image
apply = Applier()
apply = apply.to(device)

# Model
resnet = InceptionResnetV1(pretrained='vggface2').eval()
resnet = resnet.to(device)

# Original image before adversarial perturbation 
ground_truth_image = load_data('./faces/input/ronald.jpg')[0][0]
ground_truth_tensor = tensorize(ground_truth_image)
ground_truth_tensor = ground_truth_tensor.to(device)
ground_truth_emb = resnet(norm(ground_truth_tensor))

# This is used to recombine all the saved tuples into a list of tuples which are in the form of:
# (ID_Number, embedding_1, embedding_2, embedding_3, 
#  embedding_4, embedding_5)
emb_loc = glob.glob('./results/exp_1/embeddings/*')
emb_loc.sort()
adv_list = []
for loc in emb_loc:
    with open(loc, 'rb') as f:
        adv_list.append(pickle.load(f))

mask_loc = glob.glob('./results/exp_1/masks/*')
mask_loc.sort()
mask_list = []
for loc in mask_loc:
    with open(loc, 'rb') as f:
        mask_list.append(pickle.load(f))

# Data gathering
# counter is for the total number of images and mask increments
counter = 0
# t_count is for the total number of images that were true
t_count = 0
# d_count is for the total number of adv faces not detected
d_count = 0
# index to iterate through mask_list
i = 0
for adv_img_list in tqdm(adv_list):
    mask_img_list = mask_list[i]
    for img_idx in range(1, 6):
        # Initialize the base case of the closest ID
        closest_id = None
        closest_dist = emb_distance(ground_truth_emb, adv_img_list[img_idx])
        
        # Testing to see if the face is detected
        adversarial_tensor = apply(ground_truth_tensor, mask_img_list[img_idx]).detach().to(device)
        adversarial_image = imagize(adversarial_tensor.cpu())
        detection_test = fr.face_locations(np.array(adversarial_image))
        counter += 1
        # if the faces isn't detected, +1 counter, skip image search
        if not detection_test:
            d_count += 1
            continue

        # Within this identity in the database
        for db_list in database:
            # Search the 5 images of that identity
            for id_img_idx in range(1, 6):
                # For the distance
                dist = emb_distance(adv_img_list[img_idx], db_list[id_img_idx])

                # If the distance is smaller than the base case
                # then it becomes the new closest ID
                if (dist < closest_dist):
                    closest_id = db_list[0]
                    closest_dist = dist

        # Checking if the closest ID is the target ID
        if (closest_id == adv_img_list[0]): 
            t_count += 1
    i += 1

# Outputs!
print("Successful attacks  ", t_count, counter, t_count/counter)
print("Faces not detected  ", d_count)
print("Unsuccessful attacks", counter - (t_count + d_count))