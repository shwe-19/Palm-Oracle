from flask import Flask, render_template, request, redirect, url_for
from model.predict import analyze_palm_image
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        image = request.files['palm_image']
        gender = request.form['gender']
        hand = request.form['hand']
        
        if (gender.lower() == 'male' and hand.lower() != 'right') or (gender.lower() == 'female' and hand.lower() != 'left'):
            error_msg = "For males, please upload a right-hand palm image; for females, a left-hand palm image."
            return render_template('upload.html', error=error_msg)
        if image:
            filename = f"{uuid.uuid4().hex}.jpg"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            return redirect(url_for('result', image_name=filename, gender=gender, hand=hand))
    return render_template('upload.html')

@app.route('/result')
def result():
    image_name = request.args.get('image_name')
    gender = request.args.get('gender')
    hand = request.args.get('hand')
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    result_data = analyze_palm_image(image_path, gender, hand)
    return render_template('result.html', image_name=image_name, result=result_data)

@app.route('/thankyou')
def thankyou():
    return render_template('thankyou.html')

if __name__ == '__main__':
    app.run(debug=True)