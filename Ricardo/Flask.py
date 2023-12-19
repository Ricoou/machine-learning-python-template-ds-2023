from flask import Flask, request, jsonify
import pickle

# Load the model
model = pickle.load(open('/workspaces/machine-learning-python-template-ds-2023/Ricardo/Advertising_model.sav', 'rb'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)
