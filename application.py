from flask import Flask,render_template,request
from tensorflow.keras.models import load_model
from predict_CMD_YT_link import predict
from predict_CMD_YT_link import download_youtube_video
import os

#load the model
model = load_model("model_VGG16_CNN.h5")
app = Flask(__name__)



@app.route('/')
def main_page():
    return render_template('main_page.html')


@app.route('/activities')
def get_predictions():
    

    frames_count = 25

    output_directory = 'Youtube_Videos'
    os.makedirs(output_directory, exist_ok = True)

    YT_link = request.args['YT_link']
    title = download_youtube_video(YT_link,output_directory)
    path =  f'{output_directory}/{title}.mp4'
    predictions = predict(path,frames_count,model)

    #get the first activity(first key in dict)
    predicted_activity = list(predictions.keys())[0]
    
    #round the probabilities for better view
    for activity,probability in predictions.items():
        predictions[activity] = round(probability,3)
    
    return render_template('prediction_results.html',predictions=predictions,predicted_activity=predicted_activity,title=title,YT_link=YT_link)
    

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    