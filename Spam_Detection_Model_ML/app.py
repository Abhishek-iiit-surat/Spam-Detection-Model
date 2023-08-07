from flask import Flask, request, render_template
from spam2 import spamModel
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('spamDetect.html', isSpam="")


@app.route('/check', methods=["POST", "GET"])
def chk():
    message = request.form['inpText']
    arr = ['Not Spam', "Spam"]
    return render_template('spamDetect.html', isSpam=arr[spamModel(message)[0]])

if __name__ == '__main__':
    app.debug = True
app.run()
