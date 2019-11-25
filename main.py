import pandas as pd
import sqlite3
from database import Database
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import RidgeCV

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


def LR(train_df, test_df, attributes, prediction):
	print('in lr')
	x_train = train_df[attributes]
	y_train = train_df[prediction]

	x_test = test_df[attributes]
	y_test = test_df[prediction]

	lr = LinearRegression(normalize=True)
	lr = lr.fit(x_train, y_train)
	pred = lr.predict(x_test)

	# lr = RidgeCV(alphas=(0.001, 0.01, 0.1, 1.0), normalize=True)
	# lr.fit(x_train, y_train)
	# pred = lr.predict(x_test)

	# Determine mean absolute error
	mae = mean_absolute_error(y_test, pred)
	# Print `mae`
	print('Mean Absolute Error: \n', mae)

	# The coefficients
	print('Coefficients: \n', attributes, lr.coef_)

	plt.scatter(y_test, pred, color='black')
	plt.plot(y_test, y_test, color='blue', linewidth=2)
	plt.xlabel('Actual WAR')
	plt.ylabel('Predicted WAR')
	plt.title('Predicted ' + prediction + ' vs. Actual')
	plt.show()

	return lr


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


def create_LR(players_df, attributes, df_cols, years_behind, years_ahead):
	players_df = multiyear_clean_df(players_df, years_behind, years_ahead, attributes)

	new_atts = []

	for i in range(1, years_behind + 1):
		for att in attributes:
			df_cols.append(str(i)+'_Year_Prev_'+att)
			new_atts.append(str(i)+'_Year_Prev_'+att)

	attributes.extend(new_atts)

	for i in range(1, years_ahead + 1):
		df_cols.append(str(i)+'_Year_Ahead_WAR')

	prediction = str(years_ahead) + '_Year_Ahead_WAR'

	train_df, test_df = create_train_set(players_df, attributes, df_cols)
	lr = LR(train_df, test_df, attributes, prediction)


def main():
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

	years_behind = 3
	years_ahead = 4

	create_LR(players_df, attributes, df_cols, years_behind, years_ahead)


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
	# basic_database_plots(players_df)


	conn.close()

main()
