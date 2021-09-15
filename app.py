from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)


model = joblib.load("model.joblib")

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/documentation_api")
def documentation():
     return render_template ('documentation.html')
 

@app.route('/wine_quality',methods=['POST']) 
def wine_quality():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    # Load model
    classifier = joblib.load("model.joblib")
    prediction = round(classifier.predict(final_features)[0], 2) 
    return render_template('index.html', prediction_text='the prediction for this wine is :{}/10'.format(prediction))

@app.route("/api", methods=["POST"])
def api():
    # Check if request has a JSON content
    if request.json:
        # Get the JSON as dictionnary
        req = request.get_json()
        # Load model
        classifier = joblib.load("model.joblib")
        # Predict
        prediction = classifier.predict([req[key] for key in req.keys()])
        # Return the result as JSON but first we need to transform the
        # result so as to be serializable by jsonify()
        prediction = [str(prediction[index] for index in range(len(prediction)))]
        return jsonify({"prediction": prediction}), 200
    return jsonify({"msg": "Error: request not correct"}),400


# launcher of the web page
if __name__ == "__main__":
    app.run(debug=True)