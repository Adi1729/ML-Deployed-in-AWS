## ML-Model-Flask-Deployment
This is a demo project to elaborate how Machine Learn Models are deployed on production using Flask API.
Backorder Prediction. 

### Prerequisites
Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

### Project Structure
This project has four major parts :
1. app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the precited value based on our model and returns it.
2. ModelConfig.py - This uses requests module to call APIs already defined in app.py and dispalys the returned value.
3. Preprocess - This folder contains the HTML template to allow user to enter employee detail and displays the predicted employee salary.
4. Train_data 
5. template
6. static

### Running the project
1. Ensure that you are in the project home directory. 

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

You should be able to view the homepage as below :
<img src="https://github.com/Adi1729/ML-Deployed-in-AWS/blob/master/backorder_api.png">


