import flask
import pickle

# Use pickle to load in the pre-trained model.
with open(f'model/survival_rate_model_xgboost.pkl', 'rb') as f:
    model= pickle.load(f)
    
app = flask.Flask(__name__, template_folder= 'templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method =='GET':
        return(flask.render_template('Survival.html'))
    
    if flask.request.method == 'POST':
        organ = flask.request.form['organ']
        donor_type = flask.request.form['donor type']
        state = flask.request.form['state']
        
        input_variabes= pd.DataFrame([[organ, organ_type, state]],
                            columns=['Organ', 'Donor_Type, State Abv'],
                            dtype= object)
        prediction= model.predict(input_variables)[0]
        
        return flask.render_template('Survival.html',
                                    original_input= {'Organ': organ,
                                                      'Donor Type': donor_type,
                                                      'State Abbreviation': state},
                                    result= prediction)
        

if __name__ == '__main__':
    app.run()