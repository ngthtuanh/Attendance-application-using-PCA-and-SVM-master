from flask import Flask
from app import views
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = views.init_app(app)

# Add routes
app.add_url_rule(rule='/', endpoint='home', view_func=views.index)
app.add_url_rule(rule='/app/', endpoint='app', view_func=views.app)
app.add_url_rule(rule='/app/name/',
                 endpoint='name',
                 view_func=views.face_recognition_App,
                 methods=['GET', 'POST'])
app.add_url_rule(rule='/video_feed',
                 endpoint='video_feed',
                 view_func=views.video_feed)

if __name__ == '__main__':
    socketio.run(app, debug=True)