FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ./

RUN python3.12 -m pip install --target "${LAMBDA_TASK_ROOT}" -r requirements.txt

COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

CMD [ "lambda_function.lambda_handler" ]
