import flask
from flask import Flask, request, jsonify
import util

app = Flask(__name__)


@app.route("/classify_image", methods=['GET', 'POST'])
def classify():
    image_data = request.form['image_data']
    print(type(image_data))
    print(image_data)
    response = flask.Response(util.classify(image_data))

    # jsonify 
    response.headers.add('Access-Control-Allow-Origin', '*')
    print(response)
    return response


if __name__ == "__main__":
    print("Starting python flask...")
    util.load_saved_model()
    app.run(port=5002)
