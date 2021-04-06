import numpy as np
from cv2 import cv2
import os
import argparse
from tensorflow.keras.models import load_model


categories = ["Biking", "Drumming", "Basketball", "Diving","Billiards","HorseRiding","Mixing","PushUps","Skiing","Swing"]

image_height = 64
image_width = 64


def predict(video, count,model):
    ''' function that return predicted probabilities for the activities in a video,
    the prediction is done every "count" slice of frames, and got the final prediction 
    by averaging all the predictions'''
    model_ouput_size = 10
    predictions = {}
    probabilities_np = np.zeros((count, model_ouput_size), dtype = np.float)
    video_reader = cv2.VideoCapture(video)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    #calculate window size by doing integer division (some little frames will be ignored)
    window = video_frames_count // count

    for frame_counter in range(count):
        #control the video position to read every "count " slice of frames
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * window)
        _ , frame = video_reader.read() 
        resized_frame = cv2.resize(frame, (image_height, image_width))
        normalized_frame = resized_frame / 255
        #predict with the model 
        probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]
        probabilities_np[frame_counter] = probabilities
    #calculate the average probabilities for the chosen frames
    average_probabilities = probabilities_np.mean(axis = 0)
    # Sorting the Probabilities
    final_probabilities_indexes = np.argsort(average_probabilities)[::-1]
    for label in final_probabilities_indexes:
        category = categories[label]
        category_probability = average_probabilities[label]
        predictions[category] = category_probability

    video_reader.release()

    return predictions

model = load_model("model_VGG16_CNN_LSTM.h5")

if __name__ == '__main__':

    frames_count = 50
    parser=argparse.ArgumentParser(description='This program predicts the human ctivity in a video \n current list Activities: \n  [Biking, Drumming, Basketball, Diving,Billiards,HorseRiding,Mixing,PushUps,Skiing,Swing] ')
    parser.add_argument('path_toVideo',help='give a video with one of those activities \n please give the path to a short video in your local disc, with good quality that contains one person doing that activity')

    args=parser.parse_args()

    predictions = predict(args.path_toVideo,frames_count,model)
    #get the first activity(first key in dict)
    activity = list(predictions.keys())[0]

    print('\n Human Activity Recognition model predicts the following: \n')
    print(f' The Activity being done in your video is mostly: {activity} ')
    print('\n The probabilities for all the activities are given as following:\n')
    for activity,probability in predictions.items():
        print(activity, '         ',round(probability,3))