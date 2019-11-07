from pybaseball import batting_stats, pitching_stats
from database import Database

DB_NAME = 'database.sqlite'


def main():
	db = Database()
	conn = db.create_connection(DB_NAME)
	c = conn.cursor()
	c.execute('DROP table if exists batting')
	conn.commit()
	c.execute('CREATE table if not exists batting(Season real, Name text, Team text, ' + 
			'Age real, G real, AB real, PA real, H real, h_1B real, h_2B real, ' +
			'h_3B real, HR real, RBI real, SB real, AVG real, BB_p real, K_p real, ' +
			'BB_K real, OBP real, SLG real, OPS real, ISO real, BABIP real, ' + 
			'GB_FB real, LD_p real, GB_p real, FB_p real, IFFB_p real, wOBA real, ' +
			'WAR real, wRC_p real)')

	conn.commit()

	for i in range(1970, 2020):

		data = batting_stats(i)

		for i, row in data.iterrows():
			d = [(row['Season'], row['Name'], row['Team'], row['Age'], row['G'], row['AB'], 
			row['PA'], row['H'], row['1B'], row['2B'], row['3B'], row['HR'], 
			row['RBI'], row['SB'], row['AVG'],
			row['BB%'], row['K%'], row['BB/K'], row['OBP'], row['SLG'], row['OPS'],
			row['ISO'], row['BABIP'], row['GB/FB'], row['LD%'], row['GB%'], 
			row['FB%'], row['IFFB%'], row['wOBA'], row['WAR'], row['wRC+'])]
			
			c.executemany('INSERT into batting VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ' + 
			'?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', d)

		conn.commit()


	conn.close()


main()
