# CFB2

In order to run, first the dataset must be made using dataset/create_team_data.ipynb, this will be stored in team_attributes.csv

Once created, models can be trained via ml/training.ipynb
- Scikit Learn models work better, probably can ignore nn training
- Models should be named elastic_model.pkl, lars_model.pkl, lassolars_model.pkl, or ridge_model.pkl

After making models run using run.py:
- python run.py --file 'elastic lars lassolars ridge' ***CHOOSE OPTIONS****

- The above command will use the average of the 4 models to predict matchups, any number of models can be used
