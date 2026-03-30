import urllib.request, os

os.makedirs("face_detector", exist_ok=True)

urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "face_detector/deploy.prototxt"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
)
print("Downloaded!")