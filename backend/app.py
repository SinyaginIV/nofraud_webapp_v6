from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello, Railway is working!"

# Все остальные маршруты временно закомментируй или убери.
