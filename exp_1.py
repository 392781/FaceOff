import pickle 
import gc
from copy import deepcopy
from adversarial_face_recognition import *

# Opens the database
# Files are stored as a list of tuples.
# 
# [(ID_Number, embedding_1, embedding_2, embedding_3, 
#   embedding_4, embedding_5, Avg_embedding), 
#   ...
# ]
with open('./database.file', 'rb') as f:
    database = pickle.load(f)

# Loads training image
print('Loading data')
input_image = load_data('./faces/input/ronald.jpg')[0]

# Use GPU if available.
device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

# Tools ----------

# Image normalization using ImageNet normalization values
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
norm = norm.to(device)

# Used to apply a mask onto an image
apply = Applier()
apply = apply.to(device)

# Model
resnet = InceptionResnetV1(pretrained='vggface2').eval()
resnet = resnet.to(device)

# 1 image to train on for every target
input_list = []
for i in range(0, 5):
    input_list.append(input_image)

# Creating embeddings of ground truths for loss function
# Creating tensors of ground truth images for training with masks
print('Creating inputs\n')
input_emb = []
input_tensors = []
for image, _ in input_list:
    input_emb.append(resnet(norm(tensorize(image).cuda())))
    input_tensors.append(tensorize(image).cuda())

# Masks for each input image that will train...
# So in this case 5 masks, for 5 target images of 1 identity
mask_list = []
for face, _ in input_list:
    mask_list.append(create_mask(face))

# Offset the mask... Not important, was made to deal with a problem that doesn't 
# happen often
for j in range(len(mask_list)):
    mask_list[j] = mask_offset(input_list[j], mask_list[j][0], mask_list[j][1])

# TRAINING
epochs = 40
for i in range(0, len(database)):
    # For each identity in the database, reset the mask list
    list_of_masks = deepcopy(mask_list)
    opt = optim.Adamax(list_of_masks, lr = 1e-1, weight_decay = 0.0001)
    # Initialize the target images of identity 'i'
    target_emb = database[i][1:6]

    # Necessary lists for training procedure
    adversarial_list = [None for v in range(len(list_of_masks))]
    embeddings = [None for v in range(len(input_tensors))]
    losses = [None for v in range(len(input_tensors))]

    # Save locations
    save_emb_loc = './results/exp_1/id_' + str(database[i][0]) + '_emb.file'
    save_masks_loc = './results/exp_1/id_' + str(database[i][0]) + '_masks.file'

    # Begin training 40 epochs on 5 images of 1 identity
    for k in tqdm(range(epochs), desc=f'IMG #: {i+1:2}'):
        for h in tqdm(range(len(adversarial_list)), desc=f'ID: {database[i][0]:5}'):

            # Apply the mask overlay onto the image
            adversarial_list[h] = apply(input_tensors[h], list_of_masks[h])
            # Calculate the adversarial embedding
            embeddings[h] = resnet(norm(adversarial_list[h]))

            # Loss function -> Maximize distance between input_emb (ground truth)
            #                  Minimize distance between target_emb
            losses[h] = (-emb_distance(embeddings[h], input_emb[h])
                                +emb_distance(embeddings[h], target_emb[h]))
            
            # Backpropagation
            losses[h].backward(retain_graph=True)
            opt.step()
            opt.zero_grad()

            # Keep the mask pixels within a certain range
            list_of_masks[h].data.clamp_(-1, 1)
    
    # Prepare for saving...
    # list_of_masks and embeddings already have all the images
    # Just need to add the identity of the images for when we gather results
    list_of_masks.insert(0, database[i][0])
    embeddings.insert(0, database[i][0])
    # Save!
    with open(save_emb_loc, 'wb') as f:
        pickle.dump(embeddings, f)
    with open(save_masks_loc, 'wb') as f:
        pickle.dump(list_of_masks, f)

    # strongly worded letter to the interpreter to clean up :)
    # I was having memory issues, this was just to be sure everything was cleaned up
    del list_of_masks
    del adversarial_list
    del embeddings
    del losses
    del target_emb
    del opt
    gc.collect()

