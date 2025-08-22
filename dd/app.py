from flask import Flask, render_template
import threading
import detection

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start():
    t = threading.Thread(target=detection.start_detection)
    t.start()
    return "Detection Started"

@app.route('/stop')
def stop():
    detection.stop_detection()
    return "Detection Stopped"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
