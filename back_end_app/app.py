from flask import Flask, request, jsonify, make_response
from flask_restx import Api, Resource, fields
from sklearn import preprocessing
from datetime import date, timedelta
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import sys
from flask_cors import CORS
import os, re


path = os.path.dirname(__file__)
# flask_app = Flask(__name__)
# app = Api(app = flask_app, 
# 		  version = "1.0", 
# 		  title = "Iris Plant identifier", 
# 		  description = "Predict the type of iris plant")

# name_space = app.namespace('prediction', description='Prediction APIs')

# model = app.model('Prediction params', 
# 				  {'sepalLength': fields.Float(required = True, 
# 				  							   description="Sepal Length", 
#     					  				 	   help="Sepal Length cannot be blank"),
# 				  'sepalWidth': fields.Float(required = True, 
# 				  							   description="Sepal Width", 
#     					  				 	   help="Sepal Width cannot be blank"),
# 				  'petalLength': fields.Float(required = True, 
# 				  							description="Petal Length", 
#     					  				 	help="Petal Length cannot be blank"),
# 				  'petalWidth': fields.Float(required = True, 
# 				  							description="Petal Width", 
#     					  				 	help="Petal Width cannot be blank")})

# %%
mvp_cluster = 3

# %%.csv')
"""
### Data Retrieval
"""

# %%
cus_data = pd.read_csv(os.path.join(path, 'input/orders.csv'), parse_dates= ['delivery_date', 'created_at' ]) #### Making sure that the dates columns are properly read
draft = pd.read_csv(os.path.join(path, 'output/cluster_output.csv'))
rest_cluster = pd.read_csv(os.path.join(path, 'output/rest_cluster.csv'))
# %%
allowedVendor = pd.DataFrame({'vendor_id':draft[draft.cluster==mvp_cluster].vendor_id.unique()})

# %%
### Data Preprocessing

# %%
cus_data = cus_data.drop(['vendor_discount_amount','delivery_time','order_accepted_time','promo_code','driver_rating','driver_accepted_time','ready_for_pickup_time','picked_up_time','delivered_time','promo_code_discount_percentage','LOCATION_NUMBER','LOCATION_TYPE', 'CID X LOC_NUM X VENDOR'],axis=1)

# %%
cus_data.rename(columns = {'item_count':'head_count', 'deliverydistance':'vendor_distance', 'preparationtime':'waiting_time','delivery_date':'booked_date',}, inplace = True)

# %%
cus_data.drop(cus_data.index[cus_data.akeed_order_id.isnull()], inplace= True)
cus_data.drop(cus_data.index[cus_data.booked_date.isnull()], inplace= True)

# %%
### Let's also check if the booked_date is greater than the created_at ###
cus_data['booked_date'],cus_data['created_at'] = np.where(cus_data['booked_date'] >= cus_data['created_at'],(cus_data['booked_date'],cus_data['created_at']),(cus_data['created_at'],cus_data['booked_date'])) 

# %%
cus_data.head(3)

# %%
cus_data.describe()

# %%
### Replacing 'Nan' values in head_count in last 7 days and last 4 weeks with 0 ###
cus_data['head_count'][cus_data['head_count'].isnull()] = 1
cus_data['waiting_time'][cus_data['waiting_time'].isnull()] = 0
cus_data['vendor_rating'][cus_data['vendor_rating'].isnull()] = 0
cus_data['is_favorite'][cus_data['is_favorite'].isnull()] = 'No'

data = {
    'tra':
    pd.read_csv(os.path.join(path, 'input/air_visit_data.csv')),
    'as':
    pd.read_csv(os.path.join(path, 'input/air_store_info.csv')),
    'hs':
    pd.read_csv(os.path.join(path, 'input/hpg_store_info.csv')),
    'ar':
    pd.read_csv(os.path.join(path, 'input/air_reserve.csv')),
    'hr':
    pd.read_csv(os.path.join(path, 'input/hpg_reserve.csv')),
    'id':
    pd.read_csv(os.path.join(path, 'input/store_id_relation.csv')),
    'tes':
    pd.read_csv(os.path.join(path, 'input/input_data.csv')),
    'hol':
    pd.read_csv(os.path.join(path, 'input/date_info.csv')).rename(columns={
        'calendar_date': 'visit_date'
    })
}

for x,y in data.items():
    print(y.head(1))

# %%
"""
### Data Creation and Culmination of Previously Generatred Data
"""

# %%
data['id'].air_store_id.unique().shape[0]   ### Reveals that none of the customers is repeated more than once

# %%
data['id'].air_store_id.head()

# %%
airVen = pd.DataFrame({'air_store_id':data['id'].air_store_id.head(cus_data.vendor_id.unique().shape[0]), 'vendor_id':cus_data.vendor_id.unique()})
airVen

# %%
aVenMap = {}
vAenMap = {}
for i in range(airVen.shape[0]):
    aVenMap[airVen.iloc[i,0]] =airVen.iloc[i,1]
    vAenMap[airVen.iloc[i,1]] =airVen.iloc[i,0]

# %%
notInc = pd.DataFrame({'air_store_id': data['id'].air_store_id.unique()[cus_data.vendor_id.unique().shape[0]:]})
notInc

# %%
for x,y in data.items():
    if ('air_store_id' in y.columns):
        print(y.shape[0])
        y = y.drop(y.index[ y.air_store_id.isin(airVen.air_store_id).ne(True)], inplace= False)
        data[x] = y
        print(y.shape[0])

# %%
for x,y in data.items():
    if ('air_store_id' in y.columns):
        for ix,iy in aVenMap.items():
            #y.air_store_id = np.where(y.air_store_id == ix,iy,y.air_store_id)
            y.loc[y["air_store_id"] == ix, 'air_store_id'] = iy
            data[x]= y

for x,y in data.items():
    print(y.head(1))

# %%
data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

for df in ['ar', 'hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(
        lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    data[df] = data[df].groupby(
        ['air_store_id', 'visit_datetime'], as_index=False)[[
            'reserve_datetime_diff', 'reserve_visitors'
        ]].sum().rename(columns={
            'visit_datetime': 'visit_date'
        })
    print(data[df].head())

# %%
data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tra'].head()

# %%
data['tes']['visit_date'] = data['tes']['id'].map(
    lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(
    lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

data['tes'].head()

# %%
y = data['tes']
x = 'tes'
if ('air_store_id' in y.columns):
        print(y.shape[0])
        y = y.drop(y.index[ y.air_store_id.isin(airVen.air_store_id).ne(True)], inplace= False)
        data[x] = y
        print(y.shape[0])
if ('air_store_id' in y.columns):
        for ix,iy in aVenMap.items():
            #y.air_store_id = np.where(y.air_store_id == ix,iy,y.air_store_id)
            y.loc[y["air_store_id"] == ix, 'air_store_id'] = iy
            data[x]= y


# %%
data['tes'].id = data['tes'].agg(lambda x: f"{x['air_store_id']}_{x['visit_date']}", axis=1)
print(y.head())

# %%
"""
### Allowing Vendors included in Most Valuable Customer Clusters
"""

# %%
for x,y in data.items():
    if ('air_store_id' in y.columns):
        print(y.shape[0])
        y = y.drop(y.index[ y.air_store_id.isin(allowedVendor.vendor_id).ne(True)], inplace= False)
        data[x] = y
        print(y.shape[0])

# %%
unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat(
    [
        pd.DataFrame({
            'air_store_id': unique_stores,
            'dow': [i] * len(unique_stores)
        }) for i in range(7)
    ],
    axis=0,
    ignore_index=True).reset_index(drop=True)

stores.head()

# %%
"""
### Data Metric Creation
"""

# %%
tmp = data['tra'].groupby(
    ['air_store_id', 'dow'],
    as_index=False)['visitors'].min().rename(columns={
        'visitors': 'min_visitors'
    })
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(
    ['air_store_id', 'dow'],
    as_index=False)['visitors'].mean().rename(columns={
        'visitors': 'mean_visitors'
    })
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(
    ['air_store_id', 'dow'],
    as_index=False)['visitors'].median().rename(columns={
        'visitors': 'median_visitors'
    })
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(
    ['air_store_id', 'dow'],
    as_index=False)['visitors'].max().rename(columns={
        'visitors': 'max_visitors'
    })
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(
    ['air_store_id', 'dow'],
    as_index=False)['visitors'].count().rename(columns={
        'visitors': 'count_observations'
    })
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
stores.head()

# %%
"""
### Model (Label Encoder) Preprocessing 
"""

# %%
stores = pd.merge(stores, data['as'], how='left', on=['air_store_id'])
lbl = preprocessing.LabelEncoder()
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])
stores.head()

# %%
data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
data['hol'].head()

# %%
train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date'])
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date'])

train = pd.merge(data['tra'], stores, how='left', on=['air_store_id', 'dow'])
test = pd.merge(data['tes'], stores, how='left', on=['air_store_id', 'dow'])

for df in ['ar', 'hr']:
    train = pd.merge(
        train, data[df], how='left', on=['air_store_id', 'visit_date'])
    test = pd.merge(
        test, data[df], how='left', on=['air_store_id', 'visit_date'])

col = [
    c for c in train
    if c not in ['id', 'air_store_id', 'visit_date', 'visitors']
]
train = train.fillna(-1)
test = test.fillna(-1)

print('Binding to float32')

for c, dtype in zip(train.columns, train.dtypes):
    if dtype == np.float64:
        train[c] = train[c].astype(np.float32)

for c, dtype in zip(test.columns, test.dtypes):
    if dtype == np.float64:
        test[c] = test[c].astype(np.float32)
        
print(train.shape, test.shape)
print(train.head())
print(test.head())

# %%
print(test.air_store_id.unique().shape[0])

# %%
train_ids = list(train['air_store_id'])
test_ids = list(test['air_store_id'])
common_ids = sorted(list(set(train_ids).intersection(test_ids)))
print(len(train_ids), len(test_ids), len(common_ids))

# %%
store_id = common_ids[0]
# plt.figure(figsize=(15,7))
# plt.plot(train.loc[train['air_store_id'] == store_id, 'visit_date'], train.loc[train['air_store_id'] == store_id, 'visitors'])
# plt.title("Visitors for = {}".format(store_id))
# plt.xlabel("date")
# plt.ylabel("visitors")
# plt.show()

# %%
train_x = train.drop(['air_store_id', 'visit_date', 'visitors'], axis=1)
train_y = np.log1p(train['visitors'].values)
test_x = test.drop(['id', 'air_store_id', 'visit_date', 'visitors'], axis=1)
print(train_x.shape, train_y.shape, test_x.shape)

# %% Seat Predictor Model
classifier = joblib.load('classifier.joblib')
seatPredictor = joblib.load(os.path.join(path,'seatPredictor.joblib'))
# %%
predict_y = seatPredictor.predict(test_x)
test['visitors'] = np.expm1(predict_y)
# Get unique air_store_id
test_id = test.air_store_id.unique()
print(test_id[0:5])
# %%
test['is_low']=np.where(test.visitors<test.max_visitors*0.1, True, False)

# @name_space.route("/")
# class MainClass(Resource):

# 	def options(self):
# 		response = make_response()
# 		response.headers.add("Access-Control-Allow-Origin", "*")
# 		response.headers.add('Access-Control-Allow-Headers', "*")
# 		response.headers.add('Access-Control-Allow-Methods', "*")
# 		return response

# 	@app.expect(model)		
# 	def post(self):
# 		try: 
# 			formData = request.json
# 			data = [val for val in formData.values()]
# 			prediction = classifier.predict(np.array(data).reshape(1, -1))
# 			types = { 0: "Iris Setosa", 1: "Iris Versicolour ", 2: "Iris Virginica"}
# 			response = jsonify({
# 				"statusCode": 200,
# 				"status": "Prediction made",
# 				"result": "The type of iris plant is: " + types[prediction[0]]
# 				})
# 			response.headers.add('Access-Control-Allow-Origin', '*')
# 			return response
# 		except Exception as error:
# 			return jsonify({
# 				"statusCode": 500,
# 				"status": "Could not make prediction",
# 				"error": str(error)
# 			})


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route("/")
def home():
    return {"message": "Hello from backend"}

@app.route("/prediction/", methods=['POST'])
def seatPost():
		try: 
			formData = request.json
			data = [val for val in formData.values()]
			prediction = classifier.predict(np.array(data).reshape(1, -1))
			types = { 0: "Iris Setosa", 1: "Iris Versicolour ", 2: "Iris Virginica"}
			response = jsonify({
				"statusCode": 200,
				"status": "Prediction made",
				"result": "The type of iris plant is: " + types[prediction[0]]
				})
			response.headers.add('Access-Control-Allow-Origin', '*')
			return response
		except Exception as error:
			return jsonify({
				"statusCode": 500,
				"status": "Could not make prediction",
				"error": str(error)
			})
	
@app.route("/seat/prediction", methods=['POST'])
def post():
        try: 
            formData = request.json
            # print(formData)
            # data = [val for val in formData.values()]
            result_row = test[test.air_store_id == test_id[int(formData['select1'])-1]].iloc[int(formData['select2'])-1]
            result_row['visit_date']=date.today()+timedelta(int(formData['select2']))
            # result_index = [x for x in result_row.index]
            # for i,x in enumerate(result_index):
            #      if x=='visitors':
            #           result_index[i]='reserve_visitors'
            result_row.rename({'visitors':'reserve_visitors'},inplace=True)
            ## For Cluster Change
            # rest_cluster_row = rest_cluster[rest_cluster.clusters==int(formData['select1'])].iloc[int(formData['select2'])-1]
            
            ## For actual index
            rest_cluster_row = rest_cluster.iloc[int(test_id[int(formData['select1'])])-1]
            
            result_row["restaurant_cluster"] = rest_cluster_row['clusters']
            desc = [x for i,x in enumerate(rest_cluster_row.index) if rest_cluster_row[x]==True and i<10]
            result_row["restaurant_cluter_type"] = ", ".join(desc)
            print(desc)
            print(result_row['visit_date'])
            result_row = result_row[1:]
            result_string = result_row.to_string().title()
            # result_string = re.sub(r"\s+", ': ', result_string)
            result_string.replace('_',' ')
            print(result_string)
        
            response = jsonify({
                "statusCode": 200,
                "status": "Prediction made",
                "result": "Information for the particular restuarant is as follows \n" + result_string,
                })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        except Exception as error:
            return jsonify({
                "statusCode": 500,
                "status": "Could not make prediction",
                "error": str(error)
            })
		
# @app.route("/upload", methods=['POST'])
# def upload():
#     file = request.files['file']
#     file.save('uploads/' + file.filename)

#     # Load the image to predict
#     img_path = f"./uploads/{file.filename}"
#     img = image.load_img(img_path, target_size=(150, 150))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x /= 255

#     loaded_model = load_model(os.path.join(path, 'model/dogs_cat_model.h5')

#     # Make the prediction
#     prediction = loaded_model.predict(x)
#     if os.path.exists(f"./uploads/{file.filename}"):
#         os.remove(f"uploads/{file.filename}")
        
#     if prediction < 0.5:
#         return jsonify({"message": "Cat"})
#     else:
#         return jsonify({"message": "Dog"})


if __name__ == '__main__':
    app.run(debug=True)
