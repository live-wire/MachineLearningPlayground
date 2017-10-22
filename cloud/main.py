from flask import Flask, request
from flask_restful import Resource, Api
import json
import subprocess



app = Flask(__name__)
api = Api(app)
from waitress import serve

app = Flask(__name__)

@app.route("/")
def hello():
 return "Hello World!"


cpu = "--"

@app.route("/home")
def home():
    part1 = "grep 'cpu ' /proc/stat | awk "
    part2 = "'{usage=($2+$4)*100/($2+$4+$5)} END {print usage "+'"%"'+"}'"
    combined = part1+part2
    with open('out-file.txt', 'w') as f:
        subprocess.call(combined, shell=True,stdout=f)
    file = open("out-file.txt", "r")
    cpu = file.read()
    return json.dumps({'VMS':'1','CPU':cpu,'JOBS':[]})

@app.route("/job")
def job():
    print "JOB QUEUED!!!"
    return json.dumps({'JOBID':12})

if __name__ == "__main__":
    #app.run()
    serve(app,host='0.0.0.0', port=5001)