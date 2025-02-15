from flask import Flask, render_template_string
import requests

app = Flask(__name__)

@app.route('/generate-dog', methods=['GET'])
def generate_dog():
    response = requests.get("https://dog.ceo/api/breeds/image/random")
    if response.status_code == 200:
        data = response.json()
        image_url = data["message"]
        return render_template_string('<img src="{{ url }}" alt="Random Dog">', url=image_url)
    else:
        return "Failed to fetch image", 500

if __name__ == '__main__':
    app.run(debug=True,port = 5050,host = '192.168.0.113')
