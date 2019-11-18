import pandas as pd
import sqlite3
from database import Database

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



def main():
	db = Database()
	conn = db.create_connection(DB_NAME)
	
	players_df = get_dataframe(conn)


	conn.close()

main()
