FROM public.ecr.aws/lambda/python:3.7

COPY requirements.txt requirements.txt
COPY scaler.save scaler.save

RUN pip3 install --upgrade pip

RUN pip install -r requirements.txt
RUN pip3 install https://raw.githubusercontent.com/alexeygrigorev/serverless-deep-learning/master/tflite/tflite_runtime-2.2.0-cp37-cp37m-linux_x86_64.whl --no-cache-dir

COPY btc_nextday_v052022.tflite btc_nextday_v052022.tflite
COPY lambda_function.py lambda_function.py

CMD [ "lambda_function.lambda_handler" ]
