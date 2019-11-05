import pandas as pd
import numpy as np
import random as r
import torch as t
import pickle
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
from adversarial_face_recognition import detect_face, Normalize
from tqdm import tqdm


r.seed(27)
tensorize = transforms.ToTensor()
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
db_buffer = 60
img_buffer = 10
img_buffer += 1

ids = pd.read_csv('./celebA/identity_CelebA.txt', sep = ' ', header = None)

names = []
db_size = 40
for i in range(0, db_buffer):
    if (i == db_size):
        break

    
    id = r.randint(1, 10177)
    while (names.count(id) >= 1):
        id = r.randint(1, 10177)
    try:
        index, val = np.where(ids == id)
        if (len(index) < 5):
            raise IndexError()
        imgs = (id,)

        for i in index[0:img_buffer]:
            imgs += (ids.at[i,0],)

        names.append(imgs)
    except IndexError as e:
        print('Too small! Continuing search...')
        db_size += 1

print(names[0][:])

resnet = InceptionResnetV1(pretrained = 'vggface2').eval()
database = []
database_img = []
db_size = 40

for i in range(0, db_buffer):
    if (i == db_size):
        break

    id_vector = (names[i][0],)
    id_img = (names[i][0],)

    img_number = 6
    for j in tqdm(range(1, img_buffer)):
        if (j == img_number):
            break

        path = './celebA/img_celeba/' + names[i][j]
        try:
            img = detect_face(path)[0]
            vector = resnet(norm(tensorize(img)))
            vector.detach_()
            id_vector += (vector,)
            id_img += (np.asarray(img),)
        except:
            img_number += 1


    
    avg = t.zeros_like(id_vector[1]).numpy()
    for j in range(1, 6):
        avg = np.add(avg, id_vector[j].numpy())

    avg = np.divide(avg, 5)
    id_vector += (t.tensor(avg),)

    database.append(id_vector)
    database_img.append(id_img)

print(database[0])


with open('./database.file', 'wb') as f:
    pickle.dump(database, f)

with open('./database_img.file', 'wb') as f:
    pickle.dump(database_img, f)

with open('./database.file', 'rb') as f:
    dump = pickle.load(f)
print(dump)