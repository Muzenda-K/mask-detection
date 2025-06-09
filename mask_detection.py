#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ultralytics import YOLO

# Load the YOLOv8 Model
model = YOLO("yolov8n.pt")

# Train the Model on Your Dataset
results = model.train(
    data="Mask Wearing.v4-raw.yolov8/data.yaml",  
    epochs=50,         
    imgsz=416,         
    batch=8,          
    name="yolov8_custom_training"  
)


# In[4]:


# Evaluate the Model
metrics = model.val()  # Evaluate the model on the validation set

# Output Metrics
print(f"mAP50-95: {metrics.box.map}")  # mAP @ 0.5-0.95 IoU
print(f"Precision: {metrics.box.p}")    # Precision for each class
print(f"Recall: {metrics.box.r}")       # Recall for each class       # Recall for each class     # Recall


# In[5]:


# Run Inference on New Images
results = model.predict(source="test_image.jpg", save=True, conf=0.5)

# Display Results
for result in results:
    result.show()  # Show the image with bounding boxes
    result.save("output_image.jpg")  # Save the output image


# In[11]:


import matplotlib.pyplot as plt
from PIL import Image


output_image = Image.open("output_image.jpg")


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10, 10))
plt.imshow(output_image)
plt.axis("off")  # Hide axes
plt.title("YOLOv8 Predictions")
plt.show()


# In[13]:


results = model.predict(source="test_image2.jpg", save=True, conf=0.5)

for result in results:
    result.show()  # Show the image with bounding boxes
    result.save("output_image.jpg")  # Save the output image


# In[14]:


output_image = Image.open("output_image.jpg")

get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10, 10))
plt.imshow(output_image)
plt.axis("off")  # Hide axes
plt.title("YOLOv8 Predictions")
plt.show()

