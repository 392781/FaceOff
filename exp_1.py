import pickle 
import gc
from copy import deepcopy
from adversarial_face_recognition import *

with open('./database.file', 'rb') as f:
    database = pickle.load(f)

print('Loading data')
input_image = load_data('./faces/input/ronald.jpg')[0]


device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
print(t.cuda.get_device_name(0))
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
norm = norm.to(device)
apply = Applier()
apply = apply.to(device)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
resnet = resnet.to(device)

try:
    resnet.cuda()
except:
    print('No cuda :(')

input_list = []
for i in range(0, 5):
    input_list.append(input_image)

print('Creating inputs\n')
input_emb = []
input_tensors = []
for image, _ in input_list:
    input_emb.append(resnet(norm(tensorize(image).cuda())))
    input_tensors.append(tensorize(image).cuda())

mask_list = []
for face, _ in input_list:
    mask_list.append(create_mask(face))

for j in range(len(mask_list)):
    mask_list[j] = mask_offset(input_list[j], mask_list[j][0], mask_list[j][1])

epochs = 40
for i in range(0, len(database)):
    list_of_masks = deepcopy(mask_list)
    opt = optim.Adamax(list_of_masks, lr = 1e-1, weight_decay = 0.0001)
    target_emb = database[i][1:6]

    adversarial_list = [None for v in range(len(list_of_masks))]
    embeddings = [None for v in range(len(input_tensors))]
    losses = [None for v in range(len(input_tensors))]

    save_emb_loc = './results/exp_1/id_' + str(database[i][0]) + '_emb.file'
    save_masks_loc = './results/exp_1/id_' + str(database[i][0]) + '_masks.file'

    # Begin training
    for k in tqdm(range(epochs), desc=f'ID: {database[i][0]:5}'):
        for h in range(len(adversarial_list)):

            adversarial_list[h] = apply(input_tensors[h], list_of_masks[h])
            embeddings[h] = resnet(norm(adversarial_list[h].cuda()))

            losses[h] = (-emb_distance(embeddings[h], input_emb[h])
                                +emb_distance(embeddings[h], target_emb[h]))
            
            losses[h].backward(retain_graph=True)
            opt.step()
            opt.zero_grad()

            list_of_masks[h].data.clamp_(-1, 1)
    list_of_masks.insert(0, database[i][0])
    embeddings.insert(0, database[i][0])
    with open(save_emb_loc, 'wb') as f:
        pickle.dump(embeddings, f)
    with open(save_masks_loc, 'wb') as f:
        pickle.dump(list_of_masks, f)

    del list_of_masks
    del adversarial_list
    del embeddings
    del losses
    del target_emb
    del opt
    gc.collect()


        

        # NEED TO SAVE ALL THE ADVERSARIAL LISTS
        # COMPARE EACH TO THE DATABASEEEEEEEEEEE
        # FIND THE CLOSEST IDENTITY 

