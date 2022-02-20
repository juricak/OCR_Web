from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap

import os
import model

app = Flask(__name__, template_folder='Template')
Bootstrap(app)

"""
Routes
"""
@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        min_width = request.form['min_width']
        max_width = request.form['max_width']
        min_height = request.form['min_height']
        max_height = request.form['max_height']
        model_predict = request.form['model_predict']
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = os.path.join('static', uploaded_file.filename)
            uploaded_file.save(image_path)
            data = model.get_prediction(min_width, max_width, min_height, max_height, model_predict, image_path)
            result = {
                'prediction_list': data['prediction_list'],
                'image': data['image']
            }
            return render_template('result.html', result = result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug = True)
