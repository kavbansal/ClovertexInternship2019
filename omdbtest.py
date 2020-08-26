import omdb
from omdb import OMDBClient

client = OMDBClient(apikey='da4dbe2c')
name = input("Enter title: ")
time = input("Enter year: ")
if time:
    print(client.get(title=name, year=time)['plot'])
else:
    print(client.get(title=name)['plot'])
