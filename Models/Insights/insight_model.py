import os
import numpy as np
import insightface
import matplotlib.pyplot as plt
import logging
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import cv2
from datetime import datetime
print('Numpy version : ', np.__version__)
print('insight face version : ', insightface.__version__)
print('Current Path : ', os.getcwd())
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class InsightModel:
    def __init__(self, source_image_path=None, target_image_path=None):
        self.source_image_path = source_image_path
        self.target_image_path = target_image_path
        self.onnX_trained_weight_path = 'inswapper_128_onnxFile/inswapper_128.onnx'

        self.model_response_directory = 'ModelResponses'
        os.makedirs(self.model_response_directory, exist_ok=True)
    def read_image(self, image_path=None):
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    def get_face_analysis(self):
        app = FaceAnalysis(
            name='buffalo_l', providers=['CUDAExecutionProvider']
        )
        # app.prepare(ctx_id=0, det_size=(640, 640))
        app.prepare(ctx_id=0, det_size=(1024, 1024))  # Example of increasing det_size

        return app

    def plot_single_face(self, faces=None, face=None, title='add your title'):
        bbox = face['bbox']
        bbox = [int(b) for b in bbox]

        plt.imshow(faces[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
        plt.title(title)
        plt.show()

    def __call__(self):
        source_image = self.read_image(image_path=self.source_image_path)
        target_image = self.read_image(image_path=self.target_image_path)
        print('source_image', type(source_image))
        print('target_image', type(target_image))

        app = self.get_face_analysis()

        # getting faces
        source_faces = app.get(source_image)
        target_faces = app.get(target_image)

        swapper = insightface.model_zoo.get_model(self.onnX_trained_weight_path,
                                                  download=False,
                                                  download_zip=False)

        print('source_faces type : ', type(source_faces), type(source_faces[0]))
        print('target_face type : ', type(target_faces), type(target_faces[0]))
        # target_faces = source_faces[2]

        # self.plot_single_face(faces=target_faces, face=target_faces[0], title='Target Face')

        res_source_faces = source_image.copy()

        for face in source_faces:
            print('res_source_faces : type : ', type(res_source_faces))
            print('face : type : ', type(face))
            print('target_face : type : ', type(target_faces))

            res_source_faces = swapper.get(res_source_faces,
                        face,
                        target_faces[0],
                        paste_back=True)
        fix , axes = plt.subplots(1,3, figsize=(20,20))
        axes[0].imshow(source_image)
        axes[0].set_title('Source Image')
        axes[0].axis('off')

        axes[1].imshow(target_image)
        axes[1].set_title('Target Image')
        axes[1].axis('off')


        axes[1].imshow(target_image)
        axes[1].set_title('Target Image')
        axes[1].axis('off')


        axes[2].imshow(res_source_faces)
        axes[2].set_title('Swapped Faces Image')
        axes[2].axis('off')

        plt.savefig(f'{self.model_response_directory}/response_{timestamp}.png')
        plt.show()
