from flask import Flask,request,render_template

app=Flask(__name__)

@app.route("/")
def hellow_world():
    return "<h1>welcome to flask app</h1>"

@app.route("/next",methods=['GET'])
def nextnumber():
    if request.method=='GET':
        if (request.args.get('num')==None):
            return render_template('entry.html')
        elif (request.args.get('num')==''):
            return "<html> INVALID INPUT</html>"
        
        else:
            number=request.args.get('num')
            next_num=int(number)+1
            return render_template('sol.html',next_num,num=number)
        
if __name__=="__main__":
    app.run()