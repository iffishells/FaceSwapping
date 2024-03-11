from Models.Insights.insight_model import InsightModel

source_image_path = 'Datasets/internal_images/Nawaz_janghir.jpg'
target_image_path = 'Datasets/internal_images/imran_khan.jpg'

if __name__ == '__main__':
    insightface_ = InsightModel(
        source_image_path=source_image_path,
        target_image_path=target_image_path)
    insightface_()
