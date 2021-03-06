import boto3
import joblib
import pandas as pd
import numpy as np
import tflite_runtime.interpreter as tflite


def lambda_handler(event, context):

    import boto3
    import botocore
    
    ACCESS_KEY_ID = "XXXXXX"
    ACCESS_SECRET_KEY = "XXXXXXX"

    s3 = boto3.client('s3',
                      aws_access_key_id=ACCESS_KEY_ID,
                      aws_secret_access_key=ACCESS_SECRET_KEY)

    # get the raw data for the features from bucket
    raw_data_obj = s3.get_object(
        Bucket='btcpricedata', Key='current_full_df.csv')

    # convert the csv file into a pandas dataframe
    current_full_df = pd.read_csv(raw_data_obj['Body'])
    # get the last row of data for making prediction
    last_row = pd.DataFrame(current_full_df.iloc[-1, :]).T

    # get the price data and conver to pandas dataframe
    price_data_obj = s3.get_object(Bucket='btcpricedata', Key='price_df.csv')
    price_df = pd.read_csv(price_data_obj['Body'])

    # get the scaler file
    scale = joblib.load('./scaler.save')

    # scale the new data and set type and shape
    X = scale.transform(last_row.iloc[:, 1:])
    X = X.astype('float32')
    X = np.reshape(X, (1, 9))


    # call the tensorflow lite function
    model_local_path = './btc_nextday_v052022.tflite'

    interpreter = tflite.Interpreter(model_path=model_local_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']

    output_details = interpreter.get_output_details()
    output_index = output_details[0]['index']

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)

    # add prediction to price_df
    price_df.iloc[-1, 3] = preds[0]
    # update the next day prices
    price_df['next_day_price'] = price_df['price'].shift(-1)

    write_data = price_df.to_csv(index=False)

    FILE_NAME = 'price_df.csv'

    s3 = boto3.resource('s3',
                        aws_access_key_id=ACCESS_KEY_ID,
                        aws_secret_access_key=ACCESS_SECRET_KEY)

    s3.Bucket('btcpricedata').put_object(Key=FILE_NAME, Body=write_data)

    return {
        'statusCode': 200,
        'body': f"Prediction succeeded: {preds[0]} for {price_df.iloc[-1, 0]} \nUpload succeeded: {FILE_NAME} has been uploaded to Amazon S3 in bucket btcpricedata"
    }
