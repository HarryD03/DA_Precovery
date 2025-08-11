import requests

url = "https://data.minorplanetcenter.net/api/wamo"
obs_list = ['P11JuVG F51']
result = requests.get(url, json=obs_list)
observations = result.json()

# The Flask endpoint can also provide the original WAMO string
result = requests.get(url, json=['string'] + obs_list)
observations = result.text
print(observations)