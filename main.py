import pandas as pd
import time
import sqlite3
from database import Database
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

DB_NAME = 'database.sqlite'

def get_dataframe(conn, cols, df_cols):
	'''
	Takes a database connection and a list of columns in the database of
	interest and returns a Pandas dataframe of the selected data. In this case,
	it will sort the values based on a players name and then in order of 
	seasons that they played.

	'''
	c = conn.cursor()

	c.execute("SELECT * FROM batting")
 
	rows = c.fetchall()

	players_df = pd.DataFrame(rows)

	players_df.columns = cols

	players_df = players_df[df_cols]

	players_df = players_df.sort_values(['Name', 'Season'], 
		ascending=[True, True])

	return players_df

def basic_database_plots(df):
	'''
	Basic function print visualizations of the database with little to no
	manipulation. Uses imported matplotlib.pyplot as plt.

	Params: df; a Pandas dataframe of the database data
	Returns: nothing
	'''
	plt.hist(df['WAR'])
	plt.xlabel('WAR')
	plt.title('Dist of WAR')
	plt.show()

	# plt.hist(df['G'])
	# plt.xlabel('Games Played')
	# plt.title('Dist of Games Played per Season')
	# plt.show()


def create_train_set(df, attributes, cols):
	'''
	Creates training and test sets using the passed pandas dataframe

	Params: df; Pandas dataframe of original data
	Return: train_df, test_df; Pandas dataframes of the training set and test
			set respectively
	'''

	data = df[cols]

	train_df = data.sample(frac=0.75, random_state=1)
	test_df = data.loc[~data.index.isin(train_df.index)]

	return train_df, test_df


def multiyear_clean_df(df, years_behind, years_ahead, attributes):
	names = df.groupby('Name').size()

	for name, count in names.items():
		if count < years_behind + 1 or count < years_ahead + 1:
			df = df[df.Name != name]

	for i in range(1, years_behind + 1):
		for att in attributes:
			df[str(i)+'_Year_Prev_'+att] = df.groupby(['Name'])[att].shift(i)

	for i in range(1, years_ahead + 1):
		df[str(i)+'_Year_Ahead_WAR'] = df.groupby(['Name'])['WAR'].shift(-i)


	df = df.dropna()

	return df

def average_WAR(players_df):
	WAR_count = {}
	WAR_avg = {}
	for i in range(16, 70):
		WAR_count[i] = [0, 0]
	for key, value in players_df.iterrows():
		WAR_count[int(value.Age)][0] += value.WAR
		WAR_count[int(value.Age)][1] += 1

	for key, value in WAR_count.items():
		if value[1] != 0:
			WAR_avg[key] = value[0] / value[1]

	return WAR_avg

def delta_method(players_df, years_behind, years_ahead, WAR_avg, prediction):
	pred = []
	y_test = []

	output = str(years_ahead) + '_Year_Ahead_WAR' 
	for key, value in players_df.iterrows():
		y_test.append(value[output])
		avg = 0
		for i in range(1, years_behind + 1):
			j = str(i) + '_Year_Prev_WAR'
			avg += value[j]
		avg += value['WAR']
		avg /= years_behind + 1

		avg_behind = 0

		for i in range(years_behind):
			avg_behind += WAR_avg[int(value['Age'] - i)]
		avg_behind /= years_behind + 1

		avg_ahead = WAR_avg[int(value['Age'] + years_ahead)]

		diff = avg_ahead - avg_behind

		pred.append(avg + diff)

	mae = mean_absolute_error(y_test, pred)
	print('Mean Absolute Error: \n', mae)

	# Determine R2
	r2 = r2_score(y_test, pred)
	print('R2: \n', r2)

	fig, ax = plt.subplots()
	plt.scatter(y_test, pred, color='black')
	plt.plot(y_test, y_test, color='blue', linewidth=2)
	plt.xlabel('Actual WAR')
	plt.ylabel('Predicted WAR')
	plt.title('Predicted ' + prediction + ' Delta Method')
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	string = 'MAE: ' + str(mae) + '\nR2: ' + str(r2)
	plt.text(0.05, 0.95, string, verticalalignment='top', horizontalalignment='left', 
		bbox=props, transform=ax.transAxes)
	plt.show()


def create_LR(train_df, test_df, attributes, prediction):
	x_train = train_df[attributes]
	y_train = train_df[prediction]

	x_test = test_df[attributes]
	y_test = test_df[prediction]

	# lr = LinearRegression(normalize=True)
	# lr = lr.fit(x_train, y_train)
	# pred = lr.predict(x_test)

	lr = RidgeCV(alphas=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0), normalize=True)
	lr.fit(x_train, y_train)
	pred = lr.predict(x_test)

	# Determine mean absolute error
	mae = mean_absolute_error(y_test, pred)
	# Print `mae`
	print('Mean Absolute Error: \n', mae)

	# Determine R2
	r2 = r2_score(y_test, pred)
	print('R2: \n', r2)

	# The coefficients
	print('Coefficients: \n', attributes, lr.coef_)

	fig, ax = plt.subplots()
	plt.scatter(y_test, pred, color='black')
	plt.plot(y_test, y_test, color='blue', linewidth=2)
	plt.xlabel('Actual WAR')
	plt.ylabel('Predicted WAR')
	plt.title('Predicted ' + prediction + ' Ridge Regression')
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	string = 'MAE: ' + str(mae) + '\nR2: ' + str(r2)
	plt.text(0.05, 0.95, string, verticalalignment='top', horizontalalignment='left', 
		bbox=props, transform=ax.transAxes)
	plt.show()

	return lr


def create_SVR(train_df, test_df, attributes, prediction, k='linear', d=3, 
			g='auto', e=0.1, c=1.0):
	start_time = time.time()
	x_train = train_df[attributes]
	y_train = train_df[prediction]

	x_test = test_df[attributes]
	y_test = test_df[prediction]

	# tuned_params = [{'kernel': ['linear'], 'epsilon': [1]}, 
	# 	{'kernel': ['rbf'], 'gamma': [1e-3, 1e-2, 1e-1, 0, 1, 10],
	# 				'C': [1, 10, 100, 1000], 'epsilon': [0.01, 0.1, 1, 5, 10],
	# 				'degree': [1]}]

	scaler = StandardScaler()
	x_train = scaler.fit_transform(x_train)
	x_test = scaler.transform(x_test)

	# The below values were found to be optimal using grid search
	k = 'rbf'
	d = 1
	c = 100
	e = 1.0
	g = 0.001
	svr = SVR(kernel=k, degree=d, C=c, gamma=g, epsilon=e)
	
	# svr = LinearSVR(epsilon=10.0, max_iter=1000)

	# svr = GridSearchCV(SVR(), tuned_params, scoring='r2')

	svr.fit(x_train, y_train)

	# print("Best parameters set found on development set:")
	# print()
	# print(svr.best_params_)

	pred = svr.predict(x_test)

	# Determine mean absolute error
	mae = mean_absolute_error(y_test, pred)
	# Print `mae`
	print('Mean Absolute Error: \n', mae)

	# Determine R2
	r2 = r2_score(y_test, pred)
	print('R2: \n', r2)

	print("SVR took --- %s seconds ---" % (time.time() - start_time))

	fig, ax = plt.subplots()
	plt.scatter(y_test, pred, color='black')
	plt.plot(y_test, y_test, color='blue', linewidth=2)
	plt.xlabel('Actual WAR')
	plt.ylabel('Predicted WAR')
	plt.title('Predicted ' + prediction + ' SVR')
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	string = 'MAE: ' + str(mae) + '\nR2: ' + str(r2)
	plt.text(0.05, 0.95, string, verticalalignment='top', horizontalalignment='left', 
		bbox=props, transform=ax.transAxes)
	plt.show()

	
def start_analysis(players_df_u, attributes, df_cols, years_behind, years_ahead):
	prediction = str(years_ahead) + '_Year_Ahead_WAR'

	players_df = multiyear_clean_df(players_df_u, years_behind, years_ahead, attributes)
	
	WAR_avg = average_WAR(players_df_u)

	new_atts = []

	for i in range(1, years_behind + 1):
		for att in attributes:
			df_cols.append(str(i)+'_Year_Prev_'+att)
			new_atts.append(str(i)+'_Year_Prev_'+att)

	attributes.extend(new_atts)

	for i in range(1, years_ahead + 1):
		df_cols.append(str(i)+'_Year_Ahead_WAR')

	train_df, test_df = create_train_set(players_df, attributes, df_cols)
	
	lr = create_LR(train_df, test_df, attributes, prediction)

	svr = create_SVR(train_df, test_df, attributes, prediction)

	delta_method(players_df, years_behind, years_ahead, WAR_avg, prediction)



def main():
	'''
	Main part of the program. Sets the columns of the database and the columns
	of the dataframe to be used in the modeling. In addtion, sets the year
	ahead to be predicted and the number of previous years to use in predicting
	the future outcome.
	'''
	db = Database()
	conn = db.create_connection(DB_NAME)

	# List of column labels in the database
	cols = ['Season', 'Name', 'Team', 'Age', 'G', 'AB', 
				'PA', 'H', '1B', '2B', '3B', 'HR', 
				'RBI', 'SB', 'AVG',
				'BB%', 'K%', 'BB/K', 'OBP', 'SLG', 'OPS',
				'ISO', 'BABIP', 'GB/FB', 'LD%', 'GB%', 
				'FB%', 'IFFB%', 'wOBA', 'WAR', 'wRC+']

	# List of columbs we actually want in the dataframe
	df_cols = ['Season', 'Name', 'Age', 'wOBA', 'WAR', 'wRC+']
	
	players_df = get_dataframe(conn, cols, df_cols)

	attributes = ['Age', 'wOBA', 'WAR', 'wRC+']

	years_behind = 4
	years_ahead = 4

	#start_analysis(players_df, attributes, df_cols, years_behind, years_ahead)


	# Next line prints the number of NULL rows per column:
	# print(players_df.isnull().sum(axis=0).tolist())

	# Turn all NULL values into the median of the non-NULL (might not be smart)
	# players_df['GB/FB'] = players_df['GB/FB'].fillna(players_df['GB/FB'].median())
	# players_df['LD%'] = players_df['LD%'].fillna(players_df['LD%'].median())
	# players_df['GB%'] = players_df['GB%'].fillna(players_df['GB%'].median())
	# players_df['FB%'] = players_df['FB%'].fillna(players_df['FB%'].median())
	# players_df['IFFB%'] = players_df['IFFB%'].fillna(players_df['IFFB%'].median())


	# Use the below command to print out some basic plots to visualize the 
	# database:
	basic_database_plots(players_df)


	conn.close()

main()
