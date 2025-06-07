from pyhive import hive
import pandas as pd
import matplotlib.pyplot as plt

# Connect to Hive via SSH tunnel

# conn = hive.Connection(host='localhost', port=11001, auth='NOSASL')
#
# conn = hive.Connection(host='https://hd-iot-twitter-cluster-1.azurehdinsight.net', port=10001, username='sshuser')

host_name = "10.3.141.44"
port = 10000
database="ncc"


def hiveconnection(host_name, port, database):
    conn = hive.Connection(host=host_name, port=port,
                           database=database, auth='NOSASL')
    cur = conn.cursor()
    cur.execute('select * from registry')
    result = cur.fetchall()
    return result
output = hiveconnection(host_name, port, database)
print(output)

# cursor = conn.cursor()
# cursor.execute("SELECT sentiment, COUNT(*) FROM twitter_sentiments GROUP BY sentiment")

# for row in cursor.fetchall():
#     print(row)
# Run Hive query