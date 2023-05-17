from flask import Flask, send_from_directory
from flask_cors import CORS

def website_update():
	app = Flask(__name__)
	CORS(app)
	@app.route('/files/<path:filename>')
	def serve_files(filename):
		return send_from_directory('/home/pi/Project/webpage', filename)
	#if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080)

website_update()
           
#def website_update:
#	app = Flask(__name__)
#	CORS(app)
#	@app.route('/files/<path:filename>')
#	def serve_files(filename):
#		return send_from_directory('/home/pi/Project/webpage', filename)
#	app.run(host='0.0.0.0', port=8080)

#website_update()
