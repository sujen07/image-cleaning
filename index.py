from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import os
import zipfile
from werkzeug.utils import secure_filename
import cluster

app = Flask(__name__)
api = Api(app)

IMAGE_FOLDER = 'uploads'
os.makedirs(IMAGE_FOLDER, exist_ok=True)
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

class FolderUpload(Resource):
    def post(self):
        if 'file' not in request.files:
            return {'message': 'No file part'}, 400

        file = request.files['file']
        if file.filename == '':
            return {'message': 'No selected file'}, 400

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['IMAGE_FOLDER'], filename)
            folder_path = os.path.join(app.config['IMAGE_FOLDER'], filename.split('.')[0])
            file.save(file_path)

            # Extract the zip file
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(app.config['IMAGE_FOLDER'])
            except zipfile.BadZipFile:
                os.remove(file_path)
                return {'message': 'Invalid zip file'}, 400

            os.remove(file_path)

            clusters = cluster.main(folder_path)
            print(clusters)
            return {'message': 'File uploaded and extracted successfully'}, 201

api.add_resource(FolderUpload, '/upload')

if __name__ == '__main__':
    app.run(debug=True)
