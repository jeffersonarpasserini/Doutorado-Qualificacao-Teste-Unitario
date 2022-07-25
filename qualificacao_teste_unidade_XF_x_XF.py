#base libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import time
import os
#transformation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler
#dimensionality reduction
from sklearn import decomposition
import umap
#feature selection
from ReliefF import ReliefF
#classifier's
#from pcc import ParticleCompetitionAndCooperation
from sklearn.tree import DecisionTreeClassifier #J48
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
#metrics
from sklearn.metrics import accuracy_score

DATASET_PATH = "/home/jeffersonpasserini/dados/ProjetosPos/via-dataset/images/"
RESULT_PATH = "/home/jeffersonpasserini/dados/ProjetosPos/Doutorado-Qualificacao-Teste-Unitario/results/"

data_filename = RESULT_PATH+"data_detailed.csv"

#carrega todas as imagens menos a imagem testada no momento
def load_data(file_exception):
    filenames = os.listdir(DATASET_PATH)
    categories = []
    img_names = []
    for filename in filenames:
        if (filename!=file_exception):
            img_names.append(filename)
            category = filename.split('.')[0]
            if category == 'clear':
                categories.append(1)
            else:
                categories.append(0)
    
    df = pd.DataFrame({
        'filename': img_names,
        'category': categories
    })

    return df

def create_model(model_type):
    
    #CNN Parameters
    IMAGE_CHANNELS=3
    POOLING = None # None, 'avg', 'max'
    
    # load model and preprocessing_function
    if model_type=='VGG16':
        image_size = (224, 224)
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input   
        model = VGG16(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='VGG19':
        image_size = (224, 224)
        from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
        model = VGG19(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='Xception':
        image_size = (299, 299)
        from tensorflow.keras.applications.xception import Xception, preprocess_input
        model = Xception(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet50':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
        model = ResNet50(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet101':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
        model = ResNet101(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='ResNet152':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
        model = ResNet152(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))    
    elif model_type=='ResNet50V2':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
        model = ResNet50V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet101V2':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet_v2 import ResNet101V2, preprocess_input
        model = ResNet101V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='ResNet152V2':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input
        model = ResNet152V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='InceptionV3':
        image_size = (299, 299)
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        model = InceptionV3(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='InceptionResNetV2':
        image_size = (299, 299)
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
        model = InceptionResNetV2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='MobileNet':
        image_size = (224, 224)
        from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
        model = MobileNet(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))       
    elif model_type=='DenseNet121':
        image_size = (224, 224)
        from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
        model = DenseNet121(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))   
    elif model_type=='DenseNet169':
        image_size = (224, 224)
        from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
        model = DenseNet169(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,)) 
    elif model_type=='DenseNet201':
        image_size = (224, 224)
        from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
        model = DenseNet201(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))         
    elif model_type=='NASNetLarge':
        image_size = (331, 331)
        from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
        model = NASNetLarge(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))         
    elif model_type=='NASNetMobile':
        image_size = (224, 224)
        from tensorflow.keras.applications.nasnet import NASNetMobile, preprocess_input
        model = NASNetMobile(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))   
    elif model_type=='MobileNetV2':
        image_size = (224, 224)
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input        
        model = MobileNetV2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='EfficientNetB0':
        image_size = (224, 224)
        from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
        model = EfficientNetB0(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB1':
        image_size = (240, 240)
        from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input
        model = EfficientNetB1(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB2':
        image_size = (260, 260)
        from tensorflow.keras.applications.efficientnet import EfficientNetB2, preprocess_input
        model = EfficientNetB2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB3':
        image_size = (300, 300)
        from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input
        model = EfficientNetB3(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB4':
        image_size = (380, 380)
        from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
        model = EfficientNetB4(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB5':
        image_size = (456, 456)
        from tensorflow.keras.applications.efficientnet import EfficientNetB5, preprocess_input
        model = EfficientNetB5(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB6':
        image_size = (528, 528)
        from tensorflow.keras.applications.efficientnet import EfficientNetB6, preprocess_input
        model = EfficientNetB6(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB7':
        image_size = (600, 600)
        from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
        model = EfficientNetB7(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))                
    else: print("Error: Model not implemented.")

    preprocessing_function = preprocess_input

    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.models import Model
    
    output = Flatten()(model.layers[-1].output)   
    model = Model(inputs=model.inputs, outputs=output)
        
    return model, preprocessing_function, image_size

def extract_features(df_extr, model, preprocessing_function, image_size):
    df_extr["category"] = df_extr["category"].replace({1: 'clear', 0: 'non-clear'}) 
           
    datagen = ImageDataGenerator(
        #rescale=1./255,
        preprocessing_function=preprocessing_function
    )
    
    total = df_extr.shape[0]
    batch_size = 4
    
    generator = datagen.flow_from_dataframe(
        df_extr, 
        DATASET_PATH, 
        x_col='filename',
        y_col='category',
        class_mode='categorical',
        target_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )
    
    features = model.predict(generator, steps=np.ceil(total/batch_size))
    
    return features


def feature_model_extract(model_type, df_img):
    
    start = time.time()
    
    #extracting features
    if model_type=='VGG16+VGG19':           
        model_type = 'VGG16'
        modelVGG16, preprocessing_functionVGG16, image_sizeVGG16 = create_model(model_type)
        features_VGG16 = extract_features(df_img, modelVGG16, preprocessing_functionVGG16, image_sizeVGG16)
            
        model_type = 'VGG19'
        modelVGG19, preprocessing_functionVGG19, image_sizeVGG19 = create_model(model_type)
        features_VGG19 = extract_features(df_img, modelVGG19, preprocessing_functionVGG19, image_sizeVGG19)
        
        #concatenate array features VGG16+VGG19
        features = np.hstack((features_VGG16,features_VGG19))
        
    elif model_type=='Xception+ResNet50':
        model_type = 'Xception'
        modelXc, preprocessing_functionXc, image_sizeXc = create_model(model_type)
        features_Xc = extract_features(df_img, modelXc, preprocessing_functionXc, image_sizeXc)
            
        model_type = 'ResNet50'
        modelRn, preprocessing_functionRn, image_sizeRn = create_model(model_type)
        features_Rn = extract_features(df_img, modelRn, preprocessing_functionRn, image_sizeRn)
        
        #concatenate array features Xception+Resnet50
        features = np.hstack((features_Xc,features_Rn))
    
    elif model_type=='MobileNet+ResNet101':
        model_type = 'MobileNet'
        modelXc, preprocessing_functionXc, image_sizeXc = create_model(model_type)
        features_Xc = extract_features(df_img, modelXc, preprocessing_functionXc, image_sizeXc)
            
        model_type = 'ResNet101'
        modelRn, preprocessing_functionRn, image_sizeRn = create_model(model_type)
        features_Rn = extract_features(df_img, modelRn, preprocessing_functionRn, image_sizeRn)
        
        #concatenate array features Xception+Resnet50
        features = np.hstack((features_Xc,features_Rn))
    
    elif model_type=='ResNet101+DenseNet169':
        model_type = 'ResNet101'
        modelXc, preprocessing_functionXc, image_sizeXc = create_model(model_type)
        features_Xc = extract_features(df_img, modelXc, preprocessing_functionXc, image_sizeXc)
            
        model_type = 'DenseNet169'
        modelRn, preprocessing_functionRn, image_sizeRn = create_model(model_type)
        features_Rn = extract_features(df_img, modelRn, preprocessing_functionRn, image_sizeRn)
        
        #concatenate array features Xception+Resnet50
        features = np.hstack((features_Xc,features_Rn))

    elif model_type=='ResNet101+DenseNet121':
        model_type = 'ResNet101'
        modelXc, preprocessing_functionXc, image_sizeXc = create_model(model_type)
        features_Xc = extract_features(df_img, modelXc, preprocessing_functionXc, image_sizeXc)
            
        model_type = 'DenseNet121'
        modelRn, preprocessing_functionRn, image_sizeRn = create_model(model_type)
        features_Rn = extract_features(df_img, modelRn, preprocessing_functionRn, image_sizeRn)
        
        #concatenate array features Xception+Resnet50
        features = np.hstack((features_Xc,features_Rn))
        
    elif model_type=='ResNet101+MobileNetV2':
        model_type = 'ResNet101'
        modelXc, preprocessing_functionXc, image_sizeXc = create_model(model_type)
        features_Xc = extract_features(df_img, modelXc, preprocessing_functionXc, image_sizeXc)
            
        model_type = 'MobileNetV2'
        modelRn, preprocessing_functionRn, image_sizeRn = create_model(model_type)
        features_Rn = extract_features(df_img, modelRn, preprocessing_functionRn, image_sizeRn)
        
        #concatenate array features Xception+Resnet50
        features = np.hstack((features_Xc,features_Rn))
        
    elif model_type=='EfficientNetB0+MobileNet':
        model_type = 'EfficientNetB0'
        modelXc, preprocessing_functionXc, image_sizeXc = create_model(model_type)
        features_Xc = extract_features(df_img, modelXc, preprocessing_functionXc, image_sizeXc)
            
        model_type = 'MobileNet'
        modelRn, preprocessing_functionRn, image_sizeRn = create_model(model_type)
        features_Rn = extract_features(df_img, modelRn, preprocessing_functionRn, image_sizeRn)
        
        #concatenate array features Xception+Resnet50
        features = np.hstack((features_Xc,features_Rn))
    
    elif model_type=='MobileNet+ResNet50':
        model_type = 'MobileNet'
        modelXc, preprocessing_functionXc, image_sizeXc = create_model(model_type)
        features_Xc = extract_features(df_img, modelXc, preprocessing_functionXc, image_sizeXc)
            
        model_type = 'ResNet50'
        modelRn, preprocessing_functionRn, image_sizeRn = create_model(model_type)
        features_Rn = extract_features(df_img, modelRn, preprocessing_functionRn, image_sizeRn)
        
        #concatenate array features Xception+Resnet50
        features = np.hstack((features_Xc,features_Rn))
    
    elif model_type=='EfficientNetB1+EfficientNetB5':
        model_type = 'EfficientNetB1'
        modelXc, preprocessing_functionXc, image_sizeXc = create_model(model_type)
        features_Xc = extract_features(df_img, modelXc, preprocessing_functionXc, image_sizeXc)
            
        model_type = 'EfficientNetB5'
        modelRn, preprocessing_functionRn, image_sizeRn = create_model(model_type)
        features_Rn = extract_features(df_img, modelRn, preprocessing_functionRn, image_sizeRn)
        
        #concatenate array features EfficientNetB1+EfficientNetB5
        features = np.hstack((features_Xc,features_Rn))
    
    else: 
        model, preprocessing_function, image_size = create_model(model_type)
        features = extract_features(df_img, model, preprocessing_function, image_size)
        
    
    end = time.time()
    
    time_feature_extration = end-start
    
    return features

def dimensinality_reduction(model_type_reduction, number_components, allfeatures, stdScaler='No'):
    
    #print("Normaliza ---> "+stdScaler)
    
    if (stdScaler == 'Yes'):
        allfeatures_Reduction = StandardScaler().fit_transform(allfeatures)
    else:
        allfeatures_Reduction = allfeatures
    
    start = time.time()
    
    if (model_type_reduction=='PCA'):
        reduction = decomposition.PCA(n_components=number_components)
        components = reduction.fit_transform(allfeatures_Reduction)
        
    elif (model_type_reduction=='UMAP'):
        reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=number_components, metric='euclidean')
        components = reducer.fit_transform(allfeatures_Reduction)
        
    elif (model_type_reduction=='None'):
        print("Processing with all features extracted... \n")
        components = allfeatures_Reduction
              
    else: print("Error: Model not implemented. \n")
        
    end = time.time()
    time_reduction = end-start
    
    return components


def classification_model(train_data, train_label, model_classifier):
    
    if (model_classifier=='J48'):
        start = time.time()
        clf = DecisionTreeClassifier()        
        clf = clf.fit(train_data,train_label)
        end = time.time()
        time_trainning = end-start
           
    elif (model_classifier=='RBF'):
        start = time.time()
        clf = SVC(kernel='rbf')
        clf = clf.fit(train_data, train_label)
        end = time.time()
        time_trainning = end-start
        
    elif (model_classifier=='LinearSVM'):
        start = time.time()
        clf = SVC(kernel="linear", C=0.025)
        clf = clf.fit(train_data, train_label)
        end = time.time()
        time_trainning = end-start
        
    elif (model_classifier=='MLP'):
        start = time.time()
        clf = MLPClassifier(random_state=1, max_iter=1000)
        clf = clf.fit(train_data, train_label)
        end = time.time()
        time_trainning = end-start
        
    elif (model_classifier=='Logistic'):
        start = time.time()
        clf = LogisticRegression(max_iter=500)
        clf = clf.fit(train_data, train_label)
        end = time.time()
        time_trainning = end-start
        
    elif (model_classifier=='RandomForest'):
        start = time.time()
        clf = RandomForestClassifier()
        clf = clf.fit(train_data, train_label)
        end = time.time()
        time_trainning = end-start
        
    elif (model_classifier=='Adaboost'):
        start = time.time()
        clf = AdaBoostClassifier()
        clf = clf.fit(train_data, train_label)
        end = time.time()
        time_trainning = end-start
    
    elif (model_classifier=='Gaussian'):
        start = time.time()
        clf = GaussianNB()
        clf = clf.fit(train_data, train_label)
        end = time.time()
        time_trainning = end-start
        
    else: print("Error: Model not implemented. \n")
        
    return clf

def treina_modelo(model_cnn, model_red, model_class, number_comp, df_modelo):
    
    #labels array
    labels = df_modelo["category"].replace({'clear': 1, 'non-clear': 0}).to_numpy().astype(int)

    #process cnn01
    features_cnn01 = feature_model_extract(model_cnn[0], df_modelo)
    features_reduction_cnn01 = dimensinality_reduction(model_red, number_comp, features_cnn01)  
    
    #process cnn02
    features_cnn02 = feature_model_extract(model_cnn[1], df_modelo)
    features_reduction_cnn02 = dimensinality_reduction(model_red, number_comp, features_cnn02)

    features = np.hstack((features_cnn01,features_cnn02))
    features_redution = np.hstack((features_reduction_cnn01,features_reduction_cnn02))
    
    clf_model = classification_model(features_redution, labels, model_class)

    return clf_model, features, features_cnn01, features_cnn02


#---- main -----
#PCC parameters
#perc_samples = 0.1
#n_knn_neighbors = 24 
#v_p_grd = 0.5
#v_delta_v=0.1
#v_max_iter=1000000


model_type_list = [['EfficientNetB1','EfficientNetB5'],['MobileNet','ResNet101'],['ResNet101','DenseNet169'],
                   ['ResNet101','DenseNet121'],['ResNet101','MobileNetV2'],['EfficientNetB0','MobileNet'],
                   ['MobileNet','ResNet50'],['Xception','ResNet50'],['VGG16','VGG19']]

model_type = ['MobileNet','ResNet50']

model_reduction_dim_list = ['PCA']
model_reduction = 'PCA'
number_reduce_components=100
scaled_feat_reduction = 'No' # Yes or No
model_classifier_list = ['RBF']
model_classifier = 'RBF'

#image list
filenames = os.listdir(DATASET_PATH)
#filenames =['clear.017.jpg','clear.033.jpg']
#filenames =['clear.017.jpg','clear.033.jpg','clear.066.jpg','clear.080.jpg','clear.131.jpg',
#            'clear.133.jpg','clear.134.jpg','nonclear.056.jpg','nonclear.063.jpg','nonclear.086.jpg',
#            'nonclear.148.jpg']


result_img = []
result_predict = []
contador=0

for img_name in filenames: #lista de arquivos a serem testados
    
    contador+=1

    #define resultado esperado
    category = img_name.split('.')
    result = []
    img_filename = []
    img_filename.append(img_name)
    
    if(category[0]=='nonclear'):
        result.append(0)
    else:
        result.append(1)
    
    #gera dataframe da imagem a ser testada
    df_teste = pd.DataFrame({
        'filename': img_filename,
        'category': result
    })
   
    #carrega todas as imagens menos a ser testada
    df = load_data(img_name)
       
    #Faz extração, redução e treina classificador
    clf, img_features_dataset, img_features_dataset_cnn01, img_features_dataset_cnn02 = treina_modelo(model_type, model_reduction, model_classifier, number_reduce_components, df)

    #labels array
    img_label = df_teste["category"].replace({'clear': 1, 'non-clear': 0}).to_numpy().astype(int)


    #Process CNN 01
    img_features_extract_cnn01 = feature_model_extract(model_type[0], df_teste)
    #agrupa features dataset with the selected image
    img_features_cnn01 = np.vstack((img_features_dataset_cnn01,img_features_extract_cnn01))   
    #redução dimensionalidade
    img_features_reduction_cnn01 = dimensinality_reduction(model_reduction, number_reduce_components, img_features_cnn01)
    
    
    
    #Process CNN02
    img_features_extract_cnn02 = feature_model_extract(model_type[1], df_teste)
    #agrupa features dataset with the selected image
    img_features_cnn02 = np.vstack((img_features_dataset_cnn02,img_features_extract_cnn02))   
    #redução dimensionalidade
    img_features_reduction_cnn02 = dimensinality_reduction(model_reduction, number_reduce_components, img_features_cnn02)
    
    
    
    #result feature extracted
    img_features_reduction = np.hstack((img_features_reduction_cnn01,img_features_reduction_cnn02))
        
    
    #pega apenas as caracteristicas da imagem desejada
    img_features = img_features_reduction[np.shape(img_features_reduction)[0]-1].reshape(1,-1)
    
    #faz a predição da imagem
    img_result = clf.predict(img_features)
    
    print("Img#"+str(contador)+" ---> "+img_name+" wait result: "+ str(result[0])+ " predict: "+str(img_result[0]))
    
    result_img.append(result[0])
    result_predict.append(img_result[0])
    if (result[0]==img_result[0]):
        result_ok = 1
    else:
        result_ok = 0
    
    with open(data_filename,"a+") as f_data:
        f_data.write(img_name+",")
        f_data.write(str(result[0])+",")
        f_data.write(str(img_result[0])+",")
        f_data.write(str(result_ok)+"\n")
                  

print("ACC score -----> " + str(accuracy_score(result_img,result_predict)))
        