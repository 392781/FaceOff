import pickle
import glob
from tqdm import tqdm, trange
from adversarial_face_recognition import *

with open('./database.file', 'rb') as f:
    database = pickle.load(f)

# Image normalization using ImageNet normalization values
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# Model
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Original image before adversarial perturbation 
ground_truth_emb = resnet(norm(tensorize(load_data('./faces/input/ronald.jpg')[0][0])))

# This is used to recombine all the saved tuples into a list of tuples which are in the form of:
# (ID_Number, embedding_1, embedding_2, embedding_3, 
#  embedding_4, embedding_5)
emb_loc = glob.glob('./results/exp_1/embeddings/*')
emb_loc.sort()
adv_list = []
for loc in emb_loc:
    with open(loc, 'rb') as f:
        adv_list.append(pickle.load(f))

# Data gathering
# counter is for the total number of images
counter = 0
# t_count is for the total number of images that were true
t_count = 0

for adv_img_list in tqdm(adv_list):
    for img_idx in range(1, 6):
        # Initialize the base case of the closest ID
        closest_id = None
        closest_dist = emb_distance(ground_truth_emb, adv_img_list[img_idx])
        
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
    counter += 1

# Outputs!
print(t_count, counter, t_count/counter)