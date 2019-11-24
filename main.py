import pandas as pd
import sqlite3
from database import Database
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import RidgeCV

DB_NAME = 'database.sqlite'

def get_dataframe(conn, cols):
	c = conn.cursor()

	c.execute("SELECT * FROM batting")
 
	rows = c.fetchall()

	players_df = pd.DataFrame(rows)

	players_df.columns = cols

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

def create_train_set_current_year(df, attributes, cols):
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


def current_year_LR(train_df, test_df, attributes):
	x_train = train_df[attributes]
	y_train = train_df['WAR']

	x_test = test_df[attributes]
	y_test = test_df['WAR']

	lr = LinearRegression(normalize=True)
	lr.fit(x_train, y_train)
	predictions = lr.predict(x_test)

	# Determine mean absolute error
	mae = mean_absolute_error(y_test, predictions)

	# Print `mae`
	print(mae)

	rrm = RidgeCV(alphas=(0.0001, 0.001, 0.01, 0.1), normalize=True)
	rrm.fit(x_train, y_train)
	predictions_rrm = rrm.predict(x_test)

	# Determine mean absolute error
	mae_rrm = mean_absolute_error(y_test, predictions_rrm)
	print(mae_rrm)



def main():
	db = Database()
	conn = db.create_connection(DB_NAME)

	# List of column labels we are using from the database
	cols = ['Season', 'Name', 'Team', 'Age', 'G', 'AB', 
				'PA', 'H', '1B', '2B', '3B', 'HR', 
				'RBI', 'SB', 'AVG',
				'BB%', 'K%', 'BB/K', 'OBP', 'SLG', 'OPS',
				'ISO', 'BABIP', 'GB/FB', 'LD%', 'GB%', 
				'FB%', 'IFFB%', 'wOBA', 'WAR', 'wRC+']
	
	players_df = get_dataframe(conn, cols)

	# Next line prints the number of NULL rows per column:
	# print(players_df.isnull().sum(axis=0).tolist())

	# List of attributes we care about from the dataset
	attributes = ['Age', 'G', 'AB', 
				'PA', 'H', '1B', '2B', '3B', 'HR', 
				'RBI', 'SB', 'AVG',
				'BB%', 'K%', 'BB/K', 'OBP', 'SLG', 'OPS',
				'ISO', 'BABIP', 'GB/FB', 'LD%', 'GB%', 
				'FB%', 'IFFB%', 'wOBA', 'wRC+']

	# Turn all NULL values into the median of the non-NULL (might not be smart)
	players_df['GB/FB'] = players_df['GB/FB'].fillna(players_df['GB/FB'].median())
	players_df['LD%'] = players_df['LD%'].fillna(players_df['LD%'].median())
	players_df['GB%'] = players_df['GB%'].fillna(players_df['GB%'].median())
	players_df['FB%'] = players_df['FB%'].fillna(players_df['FB%'].median())
	players_df['IFFB%'] = players_df['IFFB%'].fillna(players_df['IFFB%'].median())

	# Use the below command to print out some basic plots to visualize the 
	# database:
	# basic_database_plot(players_df)

	# Use the below for predictions of the current year WAR using only the stats
	# from that year:
	# train_df, test_df = create_train_set_current_year(players_df, attributes, cols)
	# current_year_LR(train_df, test_df, attributes)

	conn.close()

main()
