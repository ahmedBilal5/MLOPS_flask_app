FROM python:3.10
# creating working directories, final models will be stored in my-model 
RUN mkdir webapp && mkdir my-model
# Environment variables are set to make the installation/exuction robust. Code doesnt need to compiled after training new models.
ENV MODEL_DIR=/webapp/my-model
ENV MODEL_FILE_LDA=clf_lda.joblib
ENV MODEL_FILE_NN=clf_nn.joblib
ENV MODEL_FILE_RFC=clf_rfc.joblib
RUN mkdir -p /webapp && mkdir /webapp/my-model
WORKDIR /webapp
COPY requirement.txt ./requirement.txt
COPY Makefile ./Makefile
COPY train.csv ./train.csv
COPY test.csv ./test.csv
COPY train.py ./train.py
COPY inference_flask.py ./app.py
# Installation of the dependecies
RUN make install
# training is executed while creating the docker image and the trained models are stored in the my-model directory
RUN python3 train.py
CMD [ "flask", "run","--host","0.0.0.0","--port","5000"]
