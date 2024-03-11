import os

import numpy as np
import insightface
import matplotlib.pyplot as plt
import logging
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


print('Numpy version : ',np.__version__)
print('insightface version : ',insightface.__version__)
import logging




# Define the log message format
log_format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create a handler for writing log messages to a file
file_handler = logging.FileHandler('logs.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)

# Create a handler for displaying log messages on the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(console_handler)

parent_path_for_saving_images = 'Datasets'
os.makedirs(f'{parent_path_for_saving_images}/input_images', exist_ok=True)
os.makedirs(f'{parent_path_for_saving_images}/output_images', exist_ok=True)

plotting_input_image = False
plotting_all_the_input_faces = False

app = FaceAnalysis(
    # name = model_name_list[0]
    # providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    name='buffalo_l', providers=['CUDAExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(640, 640))

input_filename = 't1'
# Reading Face
img = ins_get_image(f'{input_filename}')
faces = app.get(img)
# Fitting image to Model
logger.info(f'Number of faces in the current image : {len(faces)}')


logger.info(f'Input image type : {type(img)}')

if plotting_input_image == True:
    logger.info('Plotting and Saving the input Image ....')
    plt.imshow(img)
    # plt.show()
    plt.savefig(f'{parent_path_for_saving_images}/input_images/{input_filename}.png')

if plotting_all_the_input_faces == True:
    logger.info('Plotting and Saving the input Image faces ....')

    fig, axes = plt.subplots(1, 6, figsize=(12, 5))

    for i, face in enumerate(faces):
        bbox = face['bbox']
        bbox = [int(b) for b in bbox]
        axes[i].imshow(img[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
        axes[i].axis('off')
    fig.savefig(f'{parent_path_for_saving_images}/input_images/{input_filename}_faces.png')


def plot_single_face(face, title='add your title'):
    logger.info(f'Plotting single face Image for : {title}')
    bbox = face['bbox']
    bbox = [int(b) for b in bbox]

    plt.imshow(img[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
    plt.title(title)
    plt.show()


swapper = insightface.model_zoo.get_model('inswapper_128_onnxFile/inswapper_128.onnx',download=False,download_zip=False)
source_face = faces[0]
target_face = faces[2]
plot_single_face(face=source_face,title='Source Face')
plot_single_face(face=target_face,title='Target Face')

swapper.get(img,source_face,target_face,paste_back=True)

plt.imshow(img)
plt.savefig('Response.png')
# plt.show()
