#machine learning / math libraries
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import tensorflow as tf

#web frameworks
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
import uvicorn
import aiohttp
import asyncio

import os
import sys
import base64 

# initialize our Starlette application and the Keras model
app = Starlette()
model = None

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

def load_model():
	# load the pre-trained ResNet50 model 
	global model
	model = ResNet50(weights="imagenet")
	global graph
	graph = tf.get_default_graph()

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict(bytes)

@app.route("/classify-url", methods = ["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict(bytes)

def predict(bytes):
	#load byte data into a stream and into PIL Image 
    img_file = Image.open(io.BytesIO(bytes))
    # preprocess the image as input for ResNet50 model
	image = prepare_image(image, target=(224, 224))
    #encoding the image in base64 to serve in HTML
    img_pil.save("img.jpg", format="JPEG")
    img_uri = base64.b64encode(open("img.jpg", 'rb').read()).decode('utf-8')

	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# classify the input image and then initialize the list
	# of predictions to return to the client
	with graph.as_default():
		preds = model.predict(image)
		results = imagenet_utils.decode_predictions(preds)
		data["predictions"] = []

		# loop over the results and add them to the list of
		# returned predictions
		for (imagenetID, label, prob) in results[0]:
			r = {"label": label, "probability": float(prob)}
			data["predictions"].append(r)

		# indicate that the request was a success
		data["success"] = True

	# return the dictionary of results and image on a new webpage
	return HTMLResponse(
        """
        <html>
            <body>
                <p> %s </p>
            </body>
        <figure class = "figure">
            <img src="data:image/png;base64, %s" class = "figure-img">
        </figure>
        </html>
        """ %(data, img_uri))

@app.route("/")
def form(request):
        return HTMLResponse(
            """
            <h1> ResNet50 on TensorFlow </h1>
            <p> Classify what is in your image </p>
            <form action="/upload" method = "post" enctype = "multipart/form-data">
                <u> Select picture to upload: </u> <br> <p>
                1. <input type="file" name="file"><br><p>
                2. <input type="submit" value="Upload">
            </form>
            <br>
            <br>
            <u> Submit picture URL </u>
            <form action = "/classify-url" method="get">
                1. <input type="url" name="url" size="60"><br><p>
                2. <input type="submit" value="Upload">
            </form>
            """)

@app.route("/form")
def redirect_to_homepage(request):
        return RedirectResponse("/")

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    if "serve" in sys.argv:
        port = int(os.environ.get("PORT", 8008)) 
        uvicorn.run(app, host = "0.0.0.0", port = port)