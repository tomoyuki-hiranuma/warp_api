FROM python:3.6

ARG project_dir=/projects/

ADD src/requirements.txt $project_dir
# ADD . $project_dir

WORKDIR $project_dir

# RUN pip install flask
RUN pip install -r requirements.txt

# CMD ["python", "app.py"]