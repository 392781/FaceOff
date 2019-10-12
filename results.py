import pickle
import glob
from tqdm import tqdm, trange
from adversarial_face_recognition import *

with open('./database.file', 'rb') as f:
    database = pickle.load(f)

norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
resnet = InceptionResnetV1(pretrained='vggface2').eval()

ground_truth_emb = resnet(norm(tensorize(load_data('./faces/input/ronald.jpg')[0][0])))

emb_loc = glob.glob('./results/exp_1/embeddings/*')
emb_loc.sort()
adv_list = []
for loc in emb_loc:
    with open(loc, 'rb') as f:
        adv_list.append(pickle.load(f))

counter = 0
t_count = 0
for adv_img_list in tqdm(adv_list):
    for img_idx in range(1, 6):
        closest_id = None
        closest_dist = emb_distance(ground_truth_emb, adv_img_list[img_idx])
        for db_list in database:
            for id_img_idx in range(1, 6):
                dist = emb_distance(adv_img_list[img_idx], db_list[id_img_idx])
                if (dist < closest_dist):
                    closest_id = db_list[0]
                    closest_dist = dist
    if (closest_id == adv_img_list[0]): 
        t_count += 1
    counter += 1

print(t_count, counter, t_count/counter)