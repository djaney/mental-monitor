import boto3
import cv2
from threading import Thread, Event
import os
import io
from PIL import Image
import random

from flask import Flask, Response

app = Flask(__name__)

global_emotions = {}


def camera_listen(stopper, emotions):
    # Specify the camera which you want to use. The default argument is '0'
    video_capture = cv2.VideoCapture(0)
    # Capturing a smaller image fçor speed purposes
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    video_capture.set(cv2.CAP_PROP_FPS, 15)

    while stopper:
        try:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            # cv2.imshow('Video', frame)
            # cv2.waitKey(1)
            #
            # # Press Esc to exit the window
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break

            # session = boto3.Session(profile_name='hackathon')
            # client = session.client('rekognition')

            client = boto3.client('rekognition')

            pil_img = Image.fromarray(frame)
            stream = io.BytesIO()
            pil_img.save(stream, format='JPEG')
            bin_img = stream.getvalue()

            response = client.detect_faces(
                Image={
                    'Bytes': bin_img,
                },
                Attributes=["ALL"]
            )

            if 'USE_API' in os.environ and os.environ['USE_API'] == '1':
                detected_emotions = {}
                for face in response['FaceDetails']:
                    for emo in face['Emotions']:
                        if emo['Type'] not in detected_emotions:
                            detected_emotions[emo['Type']] = []
                        detected_emotions[emo['Type']].append(emo['Confidence'])

                for k, d_emo in detected_emotions.items():
                    emotions[k] = sum(d_emo) / len(d_emo) / 100

                app.logger.debug("Faces detected {}".format(len(response['FaceDetails'])))
            else:
                app.logger.debug("Using dummy data")
                for label in ['ANGRY', 'CALM', 'CONFUSED','DISGUSTED','HAPPY','SAD','SURPRISED']:
                    if label not in global_emotions:
                        global_emotions[label] = random.uniform(0, 1)
                    else:
                        global_emotions[label] += random.uniform(0, 0.1)
                        if global_emotions[label] <= 0 or global_emotions[label] >= 1:
                            global_emotions[label] = random.uniform(0, 1)

            app.logger.debug(emotions)

            # cv2.destroyAllWindows()
        except Exception as e:
            app.logger.error(str(e))


# Start thread
stopper = Event()
t = Thread(target=camera_listen, args=[stopper, global_emotions])
t.start()


@app.route('/metrics')
def metrics():
    if global_emotions:
        content = "# HELP face_emotions_type_confidence Candidates\n"
        content += "# TYPE face_emotions_type_confidence gauge\n"
        for type, confidence in global_emotions.items():
            content += "face_emotions_type_confidence{{type=\"{}\"}} {}\n".format(type, confidence)

        resp = Response(content)
        resp.headers['Content-Type'] = 'text/plain'
        return resp
    else:
        return '', 404
