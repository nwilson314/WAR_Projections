import pandas as pd
import sqlite3
from database import Database
import matplotlib.pyplot as plt

DB_NAME = 'database.sqlite'

def get_dataframe(conn):
	c = conn.cursor()

	c.execute("SELECT * FROM batting")
 
	rows = c.fetchall()

	players_df = pd.DataFrame(rows)
	cols = ['Season', 'Name', 'Team', 'Age', 'G', 'AB', 
				'PA', 'H', '1B', '2B', '3B', 'HR', 
				'RBI', 'SB', 'AVG',
				'BB%', 'K%', 'BB/K', 'OBP', 'SLG', 'OPS',
				'ISO', 'BABIP', 'GB/FB', 'LD%', 'GB%', 
				'FB%', 'IFFB%', 'wOBA', 'WAR', 'wRC+']

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

def create_training_test_sets(df):
	'''
	Creates training and test sets using the passed pandas dataframe

	Params: df; Pandas dataframe of original data
	Return: train_df, test_df; Pandas dataframes of the training set and test
			set respectively
	'''

	cols = ['Age', 'G', 'AB', 
				'PA', 'H', '1B', '2B', '3B', 'HR', 
				'RBI', 'SB', 'AVG',
				'BB%', 'K%', 'BB/K', 'OBP', 'SLG', 'OPS',
				'ISO', 'BABIP', 'GB/FB', 'LD%', 'GB%', 
				'FB%', 'IFFB%', 'wOBA', 'wRC+']

	data = df[cols]

	train_df = data.sample(frac=0.75, random_state=1)
	test_df = data.loc[~data.index.isin(train_df.index)]

	return train_df, test_df

def main():
	db = Database()
	conn = db.create_connection(DB_NAME)
	
	players_df = get_dataframe(conn)
	# Next line prints the number of NULL rows per column
	# print(players_df.isnull().sum(axis=0).tolist())

	# Turn all NULL values into the median of the non-NULL (might not be smart)
	players_df['GB/FB'] = players_df['GB/FB'].fillna(players_df['GB/FB'].median())
	players_df['LD%'] = players_df['LD%'].fillna(players_df['LD%'].median())
	players_df['GB%'] = players_df['GB%'].fillna(players_df['GB%'].median())
	players_df['FB%'] = players_df['FB%'].fillna(players_df['FB%'].median())
	players_df['IFFB%'] = players_df['IFFB%'].fillna(players_df['IFFB%'].median())

	# Use the below command to print out some basic plots to visualize the 
	# database
	# basic_database_plot(players_df)

	train_df, test_df = create_training_test_sets(players_df)

	conn.close()

main()
