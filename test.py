
#from predict_potong_flask import predict_location
import predict_potong_flask
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict_potong_current_location/<bus_line>/<bus_id>', methods=['GET'])
def predict_location(bus_line, bus_id):
    return jsonify(predict_potong_flask.predict_location(bus_line, bus_id))


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 8001


    app.run(host='0.0.0.0', port=port, debug=True)