from flask import Flask,request,render_template
import numpy as np
import pickle


model=pickle.load(open('model/model.pkl','rb'))


app= Flask(__name__)

def preprocess(input_str):
    input_list=input_str.split(',')
    input_arr=np.array(input_list, dtype=np.float32)
    return input_arr.reshape(1,-1)


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    input_string=request.form['features']
    input_array=preprocess(input_string)
    prediction=model.predict(input_array)[0]
 

    output=f"the sales value prediction is {prediction:.2f}"

    return render_template('index.html',message=output)

if __name__=="__main__":
    app.run(debug=True)
   

