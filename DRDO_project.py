#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-learn')
import os
import face_recognition
from sklearn.cluster import DBSCAN
import numpy as np


# In[2]:


# Folder containing the images
folder_path = r"C:\Users\91701\Desktop\eyed\Face-Clustering\dataset"

# Get the list of image files in the folder
image_filenames = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.jpeg', '.png'))]

# Initialize arrays to store face encodings and corresponding image filenames
encodings = []
filenames = []


# In[3]:


for filename in image_filenames:
    image = face_recognition.load_image_file(filename)
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) > 0:
        encoding = face_recognition.face_encodings(image, face_locations)[0]
        encodings.append(encoding)
        filenames.append(filename)


# In[4]:


# Convert the list of face encodings to a numpy array
encodings = np.array(encodings)

# Perform face clustering using DBSCAN algorithm
clustering = DBSCAN(metric="euclidean", n_jobs=-1)
clustering.fit(encodings)


# In[5]:


get_ipython().system('pip install matplotlib')
import matplotlib.pyplot as plt
# Retrieve the labels assigned to each face encoding
labels = clustering.labels_

# Create a dictionary to store image filenames for each cluster label
clusters = {}
for label, filename in zip(labels, filenames):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(filename)

# Visualize the face clusters
fig, axs = plt.subplots(len(clusters), figsize=(8, 8 * len(clusters)))
for i, (label, filenames) in enumerate(clusters.items()):
    axs[i].set_title(f"Cluster {label}")
    axs[i].axis("off")
    for filename in filenames:
        image = plt.imread(filename)
        axs[i].imshow(image)
plt.tight_layout()
plt.show()


# In[6]:


clusters


# In[7]:


labels


# In[10]:


from PIL import Image


# In[13]:


# Display the images in each cluster as a collage
for label, filenames in clusters.items():
    print(f"Cluster {label}: {filenames}")
    images = [Image.open(filename) for filename in filenames]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    collage = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for image in images:
        collage.paste(image, (x_offset, 0))
        x_offset += image.width

    plt.figure()
    plt.imshow(collage)
    plt.axis("off")
    plt.title(f"Cluster {label}")
    plt.show()


# In[ ]:




