import os
from google.cloud import bigquery 
from google.oauth2 import service_account 
import pandas_gbq
import pandas as pd 
import re


# get the parent directory of the current file
LEO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

#BQ Config
key_path = LEO_DIR+r"/data/cfg/project_lion_BQ.json".format(os.getcwd())
credentials = service_account.Credentials.from_service_account_file(key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],)
client = bigquery.Client(credentials = credentials, project=credentials.project_id,)


#SQL Query
#You can either used these parameters to type out the query or just write out an SQL manually.
#Note: FROM Statement must be in the form {credentials.project_id}.discord.{server_name}.
    # Leave it blank and place the desired server in tthe server name parameter.
server_name = 'talentDAO' #@param {type:"string"}
SQL_Query_SELECT = 'SELECT *' #@param {type:"string"}
SQL_Query_WHERE = "WHERE username = 'k3nn.eth'" #@param {type:"string"}

# Query1: select all from the talentDAO server
query1 = \
f"""SELECT
    username
    , message_content
    , mentions
    , channel_name
    , time_stamp
--  , CAST(time_stamp AS DATE) time_stamp
FROM {credentials.project_id}.discord.{server_name}
ORDER BY time_stamp ASC;
"""

df1 = pandas_gbq.read_gbq(query1, project_id = credentials.project_id, credentials = credentials)

df1.to_csv(LEO_DIR+f'/data/csv/{server_name}.csv', index=False)
# # Query2: select k3nn.eth messages
# query2 = \
# f"""{SQL_Query_SELECT}
# FROM {credentials.project_id}.discord.{server_name}
# {SQL_Query_WHERE};
# """
# df2 = pandas_gbq.read_gbq(query2, 
#                           project_id = credentials.project_id, 
#                           credentials = credentials, 
#                           )

#print info on querys for import
print('Importing data from the following queries:\n\n', query1)
print('Columns:\n', df1.columns)
print()
# print(query2)
# print('Columns:\n', df2.columns)