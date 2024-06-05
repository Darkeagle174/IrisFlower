from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import base64

app = Flask(__name__)

# Load the Iris dataset and train the model
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

feature_ranges = [(X[:, i].min(), X[:, i].max()) for i in range(X.shape[1])]

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

setosa_base64 = image_to_base64('iris-setosa.png')
versicolor_base64 = image_to_base64('iris-versicolor.png')
virginica_base64 = image_to_base64('iris-virginica.png')


images = {
    'setosa': 'data:image/jpeg;base64,' + setosa_base64,
    'versicolor': 'data:image/jpeg;base64,' + versicolor_base64,
    'virginica': 'data:image/jpeg;base64,' + virginica_base64
}

@app.route('/')
def home():
    return render_template('index.html', feature_ranges=feature_ranges)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = np.array([
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    ]).reshape(1, -1)
    prediction = knn.predict(user_input)
    pred_class = iris.target_names[prediction[0]]

    image_url = images[pred_class]

    return jsonify({'prediction': pred_class, 'image_url': image_url})

if __name__ == '__main__':
    app.run(debug=True)
