from tutorial.scripts.crossmodal_analysis import models, data_loading
import h5py
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from keras import optimizers

def exp_title_abs(granularity, n_papers):
    if (granularity == "2clusters"):
        num_class = 2
        dataset = "title_abstract_2clusters.h5"
        exp_weights = "title_abstract_2clusters_weights.h5"
        target_names = ['Health','Tech']
    elif (granularity == "5class"):
        num_class = 5
        dataset = "title_abstract_5class.h5"
        exp_weights = "title_abstract_5class_weights.h5"
        target_names = ['Medical and Health Sciences', 'Information and Computing Sciences', 'Engineering', 'Mathematical Sciences', 'Biological Sciences']
    else:
        print("Error calling method. Options: '2clusters' or '5class'")
        return

    batchSize= 32
    #n_papers = 4694
    print ("Number of images: "+str(n_papers))
    print ("Number of classes: "+str(num_class))

    test = list(range(0,n_papers))

    model = models.generateTextualModel(num_class, 157206)
    model.load_weights(exp_weights)
      
    db = h5py.File(dataset, "r")
    labels_test = db["labels"][test,:]
    db.close()
      
    pred = model.predict_generator(data_loading.gen_text(dataset, test, batchSize=batchSize,shuffle=False), steps = len(test)//batchSize, verbose=1) 
    maximos = np.argmax(pred,axis=1)
    predNew = np.zeros(np.shape(pred))
    for i in range(len(predNew)):
        predNew[i,maximos[i]]=1
    print(classification_report(labels_test[0:batchSize*(len(test)//batchSize)], predNew, digits=4, target_names = target_names))

def exp_captions(granularity, modality, n_captions, training=False):
    if (granularity == "2clusters"):
        num_class = 2
        target_names = ['Health','Tech']
        if (modality == "unimodal"):
            dataset = "captions_2clusters.h5"
            exp_weights = "captions_2clusters_weights.h5"
        elif (modality == "crossmodal"):
            dataset = "captions_2clusters_cross.h5"
            exp_weights = "captions_2clusters_cross_weights.h5"
        else:
            print("Error calling method. Options: 'unimodal' or 'crossmodal'")
            return
    elif (granularity == "5class"):
        num_class = 5
        target_names = ['Medical and Health Sciences', 'Engineering', 'Biological Sciences', 'Mathematical Sciences', 'Information and Computing Sciences']
        if (modality == "unimodal"):
            dataset = "captions_5class.h5"
            exp_weights = "captions_5class_weights.h5"
        elif (modality == "crossmodal"):
            dataset = "captions_5class_cross.h5"
            exp_weights = "captions_5class_cross_weights.h5"
        else:
            print("Error calling method. Options: 'unimodal' or 'crossmodal'")
            return
    else:
        print("Error calling method. Options: '2clusters' or '5class'")
        return
    if(training == False):
        batchSize= 32
        #n_captions = 8239
        print ("Number of images: "+str(n_captions))
        print ("Number of classes: "+str(num_class))

        test = list(range(0,n_captions))

        model = models.generateTextualModel(num_class, 104331)
        model.load_weights(exp_weights)

        db = h5py.File(dataset, "r")
        labels_test = db["labels"][test,:]
        db.close()
		  
        pred = model.predict_generator(data_loading.gen_text(dataset, test, batchSize=batchSize,shuffle=False), steps = len(test)//batchSize, verbose=1) 
        maximos = np.argmax(pred,axis=1)
        predNew = np.zeros(np.shape(pred))
        for i in range(len(predNew)):
	        predNew[i,maximos[i]]=1
        print(classification_report(labels_test[0:batchSize*(len(test)//batchSize)], predNew, digits=4, target_names=target_names))
    if(training == True):
        kfold = KFold(n_splits=10, shuffle=True)
        batchSize= 32
        n_captions = 1000
        print ("Number of images: "+str(n_captions))
        print ("Number of classes: "+str(num_class))

        for train, test in kfold.split([None] * n_captions):
	        model = models.generateTextualModel(num_class, 104331)
			
	        model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['categorical_accuracy'])
	        model.fit_generator(data_loading.gen_text(dataset, train, batchSize=batchSize,shuffle=True), 
                      validation_data=data_loading.gen_text(dataset, test, batchSize=batchSize,shuffle=False), 
                      steps_per_epoch = len(train)//batchSize, 
                      validation_steps = len(test)//batchSize, 
                      epochs=5)
					  
	        db = h5py.File(dataset, "r")
	        labels_test = db["labels"][test,:]
	        db.close()
			  
	        pred = model.predict_generator(data_loading.gen_text(dataset, test, batchSize=batchSize,shuffle=False), steps = len(test)//batchSize, 
					       verbose=1) 
	        maximos = np.argmax(pred,axis=1)
	        predNew = np.zeros(np.shape(pred))
	        for i in range(len(predNew)):
		         predNew[i,maximos[i]]=1
	        print(classification_report(labels_test[0:batchSize*(len(test)//batchSize)], predNew, digits=4, target_names=target_names))
			
	        break
	
def exp_figures(granularity, modality, n_images, training=False):
    if (granularity == "2clusters"):
        num_class = 2
        target_names = ['Health', 'Tech']
        if (modality == "unimodal"):
            dataset = "figures_2clusters.h5"
            exp_weights = "figures_2clusters_weights.h5"
        elif (modality == "crossmodal"):
            dataset = "figures_2clusters_cross.h5"
            exp_weights = "figures_2clusters_cross_weights.h5"
        else:
            print("Error calling method. Options: 'unimodal' or 'crossmodal'")
            return
    elif (granularity == "5class"):
        num_class = 5
        target_names = ['Medical and Health Sciences', 'Engineering', 'Biological Sciences', 'Mathematical Sciences', 'Information and Computing Sciences']
        if (modality == "unimodal"):
            exp_weights = "figures_5class_weights.h5"
            dataset = "figures_5class.h5"
        elif (modality == "crossmodal"):
            exp_weights = "figures_5class_cross_weights.h5"
            dataset = "figures_5class_cross.h5"
        else:
            print("Error calling method. Options: 'unimodal' or 'crossmodal'")
            return
    else:
        print("Error calling method. Options: '2clusters' or '5class'")
        return
    if(training == False):	
        batchSize = 32
        #n_images = 8239
		
        print ("Number of images: "+str(n_images))
        print ("Number of classes: "+str(num_class))

        test = list(range(0,n_images))
		
        model = models.generateVisualModel(num_class)
        model.load_weights(exp_weights)
		  
        db = h5py.File(dataset, "r")
        labels_test = db["labels"][test,:]
        db.close()
		  
        pred = model.predict_generator(data_loading.gen_images(dataset, test, batchSize=batchSize, shuffle=False), steps = len(test)//batchSize, verbose=1) 
        maximos = np.argmax(pred,axis=1)
        predNew = np.zeros(np.shape(pred))
        for i in range(len(predNew)):
	          predNew[i,maximos[i]]=1
        print(classification_report(labels_test[0:batchSize*(len(test)//batchSize)], predNew, digits=4, target_names=target_names))
    if(training == True):
        kfold = KFold(n_splits=10, shuffle=True)
        batchSize= 32
        n_images = 1000
		
        print ("Number of images: "+str(n_images))
        print ("Number of classes: "+str(num_class))
		
        for train, test in kfold.split([None] * n_images):
	        model = models.generateVisualModel(num_class)
			
	        adam = optimizers.Adam(lr=1e-4,decay=1e-5)
	        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

	        model.fit_generator(data_loading.gen_images(dataset, train, batchSize=batchSize,shuffle=True), 
                      validation_data=data_loading.gen_images(dataset, test, batchSize=batchSize,shuffle=False), 
                      steps_per_epoch = len(train)//batchSize, 
                      validation_steps = len(test)//batchSize, 
                      epochs=6)
					  
	        db = h5py.File(dataset, "r")
	        labels_test = db["labels"][test,:]
	        db.close()
			  
	        pred = model.predict_generator(data_loading.gen_images(dataset, test, batchSize=batchSize, shuffle=False), steps = len(test)//batchSize) 
	        maximos = np.argmax(pred,axis=1)
	        predNew = np.zeros(np.shape(pred))
	        for i in range(len(predNew)):
		        predNew[i,maximos[i]]=1
	        print(classification_report(labels_test[0:batchSize*(len(test)//batchSize)], predNew, digits=4, target_names=target_names))
			
	        break

def exp_cross(n_images, training):
    num_class = 2
    dataset = "cross.h5"
    exp_weights = "cross_weights.h5"
    if(training == False):
        n_images = 26607
        batchSize = 32
		
        print ("Number of images: "+str(n_images))
        print ("Number of classes: "+str(num_class))

        test = list(range(0,n_images))

        model = models.generateCrossModel(num_class, 104331)
        model.load_weights(exp_weights)
		  
        db = h5py.File(dataset, "r")
        labels_test = db["labels"][test,:]
        db.close()
		  
        pred = model.predict_generator(data_loading.gen_cross(dataset, test, batchSize=batchSize, shuffle=False), steps = len(test)//batchSize, verbose=1) 
        maximos = np.argmax(pred,axis=1)
        predNew = np.zeros(np.shape(pred))
        for i in range(len(predNew)):
	          predNew[i,maximos[i]]=1
        print(classification_report(labels_test[0:batchSize*(len(test)//batchSize)], predNew, digits=4))
    if(training == True):
        kfold = KFold(n_splits=10, shuffle=True)
        n_images = 1000
        batchSize = 32
		
        print ("Number of images: "+str(n_images))
        print ("Number of classes: "+str(num_class))

        for train, test in kfold.split([None] * n_images):
	        model = models.generateCrossModel(num_class, 104331)

	        adam = optimizers.Adam(lr=1e-4,decay=1e-5)
	        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
 
	        model.fit_generator(data_loading.gen_cross(dataset, train, batchSize=batchSize,shuffle=True), 
                      validation_data=data_loading.gen_cross(dataset, test, batchSize=batchSize,shuffle=False), 
                      steps_per_epoch = len(train)//batchSize, 
                      validation_steps = len(test)//batchSize, 
                      epochs=4)
					  
	        db = h5py.File(dataset, "r")
	        labels_test = db["labels"][test,:]
	        db.close()
			  
	        pred = model.predict_generator(data_loading.gen_cross(dataset, test, batchSize=batchSize, shuffle=False), steps = len(test)//batchSize, verbose=1) 
	        maximos = np.argmax(pred,axis=1)
	        predNew = np.zeros(np.shape(pred))
	        for i in range(len(predNew)):
		        predNew[i,maximos[i]]=1
	        print(classification_report(labels_test[0:batchSize*(len(test)//batchSize)], predNew, digits=4))
		
	        break
