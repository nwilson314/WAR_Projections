import sqlite3

class Database:
	def __init__(self):
		pass

	def create_connection(self, db_file):
		""" create a database connection to the SQLite database
		specified by the db_file
		:param db_file: database file
		:return: Connection object or None
		"""
		conn = None
		try:
			conn = sqlite3.connect(db_file)
		except Error as e:
			print(e)

		return conn
