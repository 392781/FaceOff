from adversarial_face_recognition import *

## Standard normalization for ImageNet images found here:
## https://github.com/pytorch/examples/blob/master/imagenet/main.py
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
apply = Applier()
## Transformations to be used later
tensorize = transforms.ToTensor()
imagize = transforms.ToPILImage()
## FaceNet PyTorch model
resnet = InceptionResnetV1(pretrained='vggface2').eval()

## Image preprocessing
input_image_location =  './faces/input/john.jpg'
target_image_location = './faces/target/nick.jpg'
input_test_location =   './faces/input_tests/john2.jpg'
target_test_location =  './faces/target_tests/nick2.jpg'

input_image = detect_face(input_image_location)[0]
print("\nInput detected and aligned")
target_image = detect_face(target_image_location)[0]
print("Target detected and aligned")
input_image.save('./results/example/input-face.png')
target_image.save('./results/example/target-face.png')

## Mask creation
mask = create_mask(input_image)[0]
delta = tensorize(mask)
delta.requires_grad_(True)

## Optimizer, some options to consider: Adam, SGD
opt = optim.Adamax([delta], lr = 1e-1, weight_decay = 0.0001)

## Initializing the FaceNet embeddings to be used in the loss function
input_emb = resnet(norm(tensorize(input_image)))
target_emb = resnet(norm(tensorize(target_image)))
print("Embeddings created")

## Will be used to combine with mask for training
input_tensor = tensorize(input_image)

## Number of training rounds
epochs = 45

## Adversarial training
## 'loss' maximizes the distance between the adversarial embedding and the
## original input embedding and minimizes the distance between the adversarial
## embedding and the target embedding
print(f'\nEpoch |   Loss   | Face Detection')
print(f'---------------------------------')
for i in range(epochs):
    adver = apply(input_tensor, delta)
    adv = imagize(adver.detach())
    embedding = resnet(norm(adver))
    loss = (-emb_distance(embedding, input_emb)
            +emb_distance(embedding, target_emb))

    ## Some pretty printing and testing to check whether face detection passes
    if i % 5 == 0 or i == epochs - 1:
        detection_test = fr.face_locations(np.array(adv))
        if not detection_test:
            d = 'Failed'
        else:
            d = 'Pass ' + str(detection_test)
        print(f'{i:5} | {loss.item():8.5f} | {d}')
        
        adv.show()

    ## Backprop step
    loss.backward(retain_graph=True)
    opt.step()
    opt.zero_grad()

    delta.data.clamp_(-1, 1)

## Additional testing image for the ground truth 
temp = detect_face(input_test_location)[0]
true_emb = resnet(norm(tensorize(temp)))
## Additional testing image for the target
temp = detect_face(target_test_location)[0]
test_emb = resnet(norm(tensorize(temp)))

## Distance calculations and "pretty" printing
print("\n target vs 2nd target  ", emb_distance(target_emb, test_emb).item())
print("\ninput img vs true img  ", emb_distance(input_emb, true_emb).item())
print("advrs img vs true img  ", emb_distance(resnet(norm(apply(input_tensor, delta))), true_emb).item())
print("\ninput img vs target    ", emb_distance(input_emb, target_emb).item())
print("advrs img vs target    ", emb_distance(resnet(norm(apply(input_tensor, delta))), target_emb).item())
print("\ninput img vs 2nd target", emb_distance(input_emb, test_emb).item())
print("advrs img vs 2nd target", emb_distance(resnet(norm(apply(input_tensor, delta))), test_emb).item())

## Final results
Image.fromarray(np.hstack(
    (np.asarray(input_image.resize((300,300))), 
     np.asarray(imagize(delta.detach()).resize((300,300))),
     np.asarray(imagize((input_tensor + delta).detach()).resize((300,300))), 
     np.asarray(target_image.resize((300,300)))))).show()
imagize(delta.detach()).save('./results/example/delta.png')
imagize((input_tensor + delta).detach()).save('./results/example/combined-face.png')