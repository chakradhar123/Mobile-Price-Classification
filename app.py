import flask
import pandas as pd
from joblib import dump, load


with open(f'./mobilePrice.joblib', 'rb') as f:
    model = load(f)
scale_cols=['battery_power',
'mobile_wt',

'px_height',
'px_width',
'ram',
]
scalers=[]
for i in range(len(scale_cols)):

    with open(f'./'+scale_cols[i]+'Scaler.bin', 'rb') as f:
        scalers.append(load(f))

app = flask.Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('main.html'))

    if flask.request.method == 'POST':
        battery_power = flask.request.form['battery_power']
        mobile_wt = flask.request.form['mobile_wt']
        px_height = flask.request.form['px_height']
        px_width = flask.request.form['px_width']
        ram = flask.request.form['ram']
        touch_screen = flask.request.form['touch_screen']

        input_variables = pd.DataFrame([[battery_power, mobile_wt, px_height, px_width, ram,
       touch_screen]],
                                       columns=['battery_power', 'mobile_wt', 'px_height', 'px_width', 'ram',
       'touch_screen'],
                                       dtype='float',
                                       index=['input'])
        for i in range(len(scale_cols)):
            m=scalers[i]
            input_variables[scale_cols[i]]=m.transform(input_variables[scale_cols[i]].values.reshape(-1,1))
        predictions = model.predict(input_variables)[0]
        print(predictions)
        res=""
        if(predictions==0):
            res="Low"
        elif(predictions==1):
            res="Medium"
        elif(predictions==2):
            res="High"
        else:
            res="Very High"
        print(touch_screen)
        return flask.render_template('main.html', original_input={battery_power:'battery_power',mobile_wt: 'mobile_wt',px_height: 'px_height', px_width:'px_width', ram:'ram',
       touch_screen:'touch_screen'},
                                     result=res)


if __name__ == '__main__':
    app.run(debug=True)