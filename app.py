from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/", methods=["POST"])
def getData():
  pass

app.run(debug=True)
