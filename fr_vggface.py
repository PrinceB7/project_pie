# Example of face detection with a vggface2 model
from numpy import expand_dims
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions


def extract_face(filename, required_size=(224, 224)):
 
    pixels = pyplot.imread(filename)
    detector = MTCNN()

    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height 
    face = pixels[y1:y2, x1:x2]
    
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


#pixels = extract_face('data/celebrity_data/train/madonna/madonna1.jpg')
pixels = extract_face('data/k3.jpg')
pixels = pixels.astype('float32')
samples = expand_dims(pixels, axis=0)
samples = preprocess_input(samples, version=2)
model = VGGFace(model='resnet50')
#print(model.summary())
yhat = model.predict(samples)
results = decode_predictions(yhat)
#print('results: ', len(results[0]))
for result in results[0]:
    print('%s: %.3f%%' % (result[0], result[1]*100))