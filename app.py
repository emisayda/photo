from flask import Flask, request, send_file
from flask_cors import CORS
from u2net_infer import load_model, predict
import os

app = Flask(__name__)
CORS(app)
model = load_model()

@app.route('/cutout', methods=['POST'])
def cutout():
    file = request.files['image']
    filepath = f'temp/{file.filename}'
    file.save(filepath)

    result = predict(model, filepath)
    output_path = f'temp/processed_{file.filename}.png'
    result.save(output_path)

    return send_file(output_path, mimetype='image/png')

if __name__ == '__main__':
    os.makedirs('temp', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
