from wsgiref import simple_server
from flask import Flask, request, render_template,send_file, Response
import os
from datetime import datetime
from api_logger import logger
from Train_data import Train
import flask_monitoringdashboard as dashboard

app = Flask(__name__)
dashboard.bind(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sample-file')
def sample_file():
    return send_file('../sample/sample.csv',
                     mimetype='text/csv',
                     attachment_filename='backorder_sample_input.csv',
                     as_attachment= True)

@app.route('/predict',methods=['POST'])
def predict():

    #Files uploaded
    f = request.files['file']

    if  len(f.filename) == 0 :
        logger.error(f'No file uploaded')
        return "No file uploaded"

    if f.filename.split('.')[1] != 'csv':
        logger.error(f'Non csv file received: {f.filename}')
        return "Uploaded non-csv file."

    try:

        logger.info(f'\n\n\n File received {f.filename}...\n')

        time_ = datetime.now().strftime("%d_%m_%H_%M")
        filename = f.filename.split('.')[0] + '_'+ time_ +'.csv'
        f.save(os.path.join('../input', filename))

        train = Train(filename)

        # Data Quality Check and imputation with proper values if required
        train.process_data()

        # Prediction
        train.predict()

        logger.info(" \n -------- SUCCESS --------")

        # Once control reaches error add suffix 'SUFFIX' to log filename

        return send_file(os.path.join('../prediction', 'output_' + filename),
                                        mimetype='text/csv',
                                        attachment_filename='backorder_prediction.csv',
                                        as_attachment= True)

    except Exception as e:
        logger.error(e,exc_info=True)
        logger.error(" \n -------- FAILURE --------")
        # Once control reaches error add suffix 'FAILURE' to log filename
        return Response("Error Occurred! %s" %e)


if __name__ == "__main__":
    #port = int(os.getenv('PORT', 5000))
    #print("Starting app on port %d" % port)
    #app.run(debug=True, port=port, host='0.0.0.0')
    host = '0.0.0.0'
    port = 5000
    httpd = simple_server.make_server(host, port, app)
    logger.info("Serving on %s %d" % (host, port))
    httpd.serve_forever()