import os
import shutil
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

sdir=r'C:\\Users\\jacks\\Downloads\\dataset'
aug_dir=os.path.join(sdir,'augmented_images')
if os.path.isdir(aug_dir): # see if aug_dir exists if so remove it to get a clean slate
    shutil.rmtree(aug_dir)
os.mkdir(aug_dir) # make a new empty aug_dir
filepaths=[]
labels=[]
# iterate through original_images and create a dataframe of the form filepaths, labels
original_images_dir=os.path.join(sdir, 'original_images')
for klass in ['helicopters']:
    os.mkdir(os.path.join(aug_dir,klass)) # make the class subdirectories in the aug_dir
    classpath=os.path.join(original_images_dir, klass) # get the path to the classes (benign and maligant)
    flist=os.listdir(classpath)# for each class the the list of files in the class    
    for f in flist:        
        fpath=os.path.join(classpath, f) # get the path to the file
        filepaths.append(fpath)
        labels.append(klass)
    Fseries=pd.Series(filepaths, name='filepaths')
    Lseries=pd.Series(labels, name='labels')
df=pd.concat([Fseries, Lseries], axis=1) # create the dataframe
gen=ImageDataGenerator(horizontal_flip=True)
groups=df.groupby('labels') # group by class
for label in df['labels'].unique():  # for every class               
    group=groups.get_group(label)  # a dataframe holding only rows with the specified label 
    sample_count=len(group)   # determine how many samples there are in this class  
    aug_img_count=0
    target_dir=os.path.join(aug_dir, label)  # define where to write the images    
    aug_gen=gen.flow_from_dataframe( group,  x_col='filepaths', y_col=None, target_size=(512,512), class_mode=None,
                                        batch_size=1, shuffle=False, save_to_dir=target_dir, save_prefix='aug-',
                                        save_format='jpg')
    while aug_img_count<len(group):
        images=next(aug_gen)            
        aug_img_count += len(images) 