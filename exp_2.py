import pickle
import glob
import PIL.Image
from adversarial_face_recognition import *
from face_recognition import *

device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
apply = Applier()
apply = apply.to(device)

def create_mask_np(mask_list, id, img):
    adversarial = apply(ground_truth_tensor, mask_list[id][img]).detach()
    masked_image = imagize(adversarial.cpu())

    try:
        adv_enc = face_encodings(np.asarray(masked_image))
        return adv_enc
    except:
        print('oopsie whoopsie')
    return 1



ground_truth_image = load_data('./faces/input/ronald.jpg')[0][0]
ground_truth_tensor = tensorize(ground_truth_image).to(device)

mask_loc = glob.glob('./results/exp_1/masks/*')
mask_loc.sort()
mask_list = []
for loc in mask_loc:
    with open(loc, 'rb') as f:
        t.load(f, lambda storage, loc: storage)
        mask_list.append(pickle.load(f))

with open('./database.file', 'rb') as f:
    t.load(f, lambda storage, loc: storage)
    database = pickle.load(f)

emb_list = []
emb_names = []
for ID in database:
    for i in range(1, 6):
        img = imagize(ID[i].detach().cpu())
        enc = face_encodings(np.asarray(img))
        emb_list.append(enc)
        emb_names.append(ID[0])

test_img = create_mask_np(mask_list, 4, 4)

print(compare_faces(emb_list, test_img))