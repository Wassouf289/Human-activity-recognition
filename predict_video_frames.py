import numpy as np
from cv2 import cv2
import os
import pafy
import argparse
from tensorflow.keras.models import load_model
from collections import deque

output_directory = 'Youtube_Videos'
os.makedirs(output_directory, exist_ok = True)


Activities = ["Biking", "Drumming", "Basketball", "Diving","Billiards","HorseRiding","Mixing","PushUps","Skiing","Swing"]

image_height = 64
image_width = 64

def predict_video_frames(video_file_path, output_file_path, window_size,model):
    ''' this function predicts the activity from a video given
    from the user,and writes the resulted video in output folder, the function uses deque object
    with the size of the number of frames we want to average on over, because we don't want to
    predict for every single frame which leads to flickering, window_size=25: it means we will average the prediction for 
    every 25 frames'''
    # Initialize a Deque Object with a fixed size 
    predicted_labels_probabilities_deque = deque(maxlen = window_size)

    video_reader = cv2.VideoCapture(video_file_path)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 24, (original_video_width, original_video_height))

    while True: 
        status, frame = video_reader.read() 
        if not status:
            break
        #resize frame
        resized_frame = cv2.resize(frame, (image_height, image_width))
        # Normalize the resized frame 
        normalized_frame = resized_frame / 255

        # make predictions
        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]

        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)

        # Assuring that the Deque is completely filled before starting the averaging process
        if len(predicted_labels_probabilities_deque) == window_size:

            # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)
            # Calculating Average of Predicted Labels Probabilities Column Wise 
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)

            # Converting the predicted probabilities into labels by returning the index of the maximum value.
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)
            activity = Activities[predicted_label]
          
            # Overlaying activity Text Ontop of the Frame
            cv2.putText(frame, activity, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Writing The Frame
        video_writer.write(frame)


        cv2.imshow('Predicted Frames', frame)

        key_pressed = cv2.waitKey(10)

        if key_pressed == ord('q'):
            break

    cv2.destroyAllWindows()

    video_reader.release()
    video_writer.release()



if __name__ == '__main__':

    model = load_model("model_VGG16_CNN_LSTM.h5")

    #number of the frames that we will average the prediction on
    window_size = 25

    parser=argparse.ArgumentParser(description='This program predicts the human ctivity in a youtube video \n current list Activities: \n  [Biking, Drumming, Basketball, Diving,Billiards,HorseRiding,Mixing,PushUps,Skiing,Swing] ')
    parser.add_argument('path_toVideo',help='give a video with one of those activities \n please give a link to a short video with good quality that contains one person doing that activity')
    args=parser.parse_args()

    output_video_file_path = f'{output_directory}/ video_HAR_CNN {window_size}.mp4'
    predict_video_frames(args.path_toVideo,output_video_file_path, window_size,model)
    
    