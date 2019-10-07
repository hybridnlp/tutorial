from keras import applications
from keras import backend as K
import cv2
import random
from tutorial.scripts.crossmodal_analysis import models, data_loading
import numpy as np
import h5py
from PIL import Image
from keras.layers.core import Lambda
import tensorflow as tf
import matplotlib.pyplot as plt

def getIndices(granularity):
    if (granularity == "2clusters"):
        num_class = 2
        dataset = "quality2clusters.h5"
        exp_weights_uni = "qualityUni2clusters.h5"
        exp_weights_mix = "qualityMix2clusters.h5"

    elif (granularity == "5class"):
        num_class = 5
        dataset = "quality5class.h5"
        exp_weights_uni = "qualityUni5class.h5"
        exp_weights_mix = "qualityMix5class.h5"

    else:
        print("Error")

    db = h5py.File("./tutorial/scripts/crossmodal_analysis/indices_quality.h5", "r")
    indices_res = list(db["indices_res"][:])
    correct_class = list(db["correct_class"][:])
    pred1 = db["pred1"][:,:]
    pred2 = db["pred2"][:,:]
    db.close()
    return indices_res, correct_class, pred1, pred2

def getCAM(granularity):
    if (granularity == "2clusters"):
        num_class = 2
        dataset = "quality2clusters.h5"
        exp_weights_uni = "qualityUni2clusters.h5"
        exp_weights_mix = "qualityMix2clusters.h5"

    elif (granularity == "5class"):
        num_class = 5
        dataset = "quality5class.h5"
        exp_weights_uni = "qualityUni5class.h5"
        exp_weights_mix = "qualityMix5class.h5"
    else:
        print("Error")
        
    indices_res, correct_class, pred1, pred2 = getIndices(granularity)
    i0 = random.randint(0,len(indices_res))
    i1 = random.randint(0,len(indices_res))    
    i2 = random.randint(0,len(indices_res))
    i3 = random.randint(0,len(indices_res))
    indices = [i0,i1,i2,i3]
    ids = [indices_res[i0],indices_res[i1],indices_res[i2],indices_res[i3]]
    for indice in indices:
        i = indices_res[indice]
        predicted_class = correct_class[indice]
        predUni = pred1[i,predicted_class]
        predMix = pred2[i,predicted_class]
        diff = predMix - predUni
        print ("Diferencia de "+str(i)+": "+str(diff*100)+"(Uni: "+str(predUni)+"; Mix: "+str(predMix)+")")
        list_img = []
        db = h5py.File(dataset, "r")
        original_img = db["images"][i,:,:,:]
        img_fr_a = Image.fromarray(original_img, 'RGB')
        img_fr_a.save("./"+str(i)+".png")
        db.close()
        list_img.append(original_img)
        img = np.array(list_img)
        cam, heatmap = grad_cam(exp_weights_uni, img, predicted_class, num_class)
        cv2.imwrite("./uni-"+str(i)+".png", cam)
        cv2.imwrite("./uni-heatmap"+str(i)+".png", heatmap)
        cam, heatmap = grad_cam(exp_weights_mix, img, predicted_class, num_class)
        cv2.imwrite("./mix-"+str(i)+".png", cam)
        cv2.imwrite("./mix-heatmap"+str(i)+".png", heatmap)
    for index in ids:
        img0 = cv2.imread("./mix-"+str(index)+".png", flags=cv2.IMREAD_COLOR)
        img1 = cv2.imread("./uni-"+str(index)+".png", flags=cv2.IMREAD_COLOR)
        img2 = cv2.imread("./5class/"+str(index)+".png", flags=cv2.IMREAD_COLOR)
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5, forward=True)
        ax2 = fig.add_subplot(131)
        ax2.set_title("Original image. ID: "+str(index))
        ax2.imshow(img2)
        ax1 = fig.add_subplot(132)
        ax1.set_title("Unimodal. ID: "+str(index))
        ax1.imshow(img0)
        ax2 = fig.add_subplot(133)
        ax2.set_title("Crossmodal. ID: "+str(index))
        ax2.imshow(img1)
          
def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)
            
def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = models.generateVisualModel(num_class)
        #new_model.load_weights('./models/modelMixHvsT1.h5')
    return new_model

def compile_saliency_function(model):
    input_img = model.input
    layer_output = model.layers[-6].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_cam(weights, image, category_index, num_class):
    input_model = models.generateVisualModel(num_class)
    input_model.load_weights(weights)
    nb_classes = num_class
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    input_model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))
    
    loss = K.sum(input_model.layers[-1].output)
    conv_output =  input_model.layers[-7].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([input_model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    visible_heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    return np.uint8(cam), visible_heatmap
