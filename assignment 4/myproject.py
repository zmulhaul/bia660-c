import os
import json
import random
import requests
import time


from flask import Flask, request, Response

# not allowing for server to exceed max tries - stackoverflow assisted
from requests.adapters import HTTPAdapter
s = requests.Session()
s.mount('http://slack.com', HTTPAdapter(max_retries=5))


application = Flask(__name__)

# FILL THESE IN WITH YOUR INFO
my_bot_name = 'zachgraysonbot' #e.g. zac_bot
my_slack_username = 'zacharymulhaul' #e.g. zac.wentzell


slack_inbound_url = 'https://hooks.slack.com/services/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'


# this handles POST requests sent to your server at SERVERIP:41953/slack
@application.route('/slack', methods=['POST'])
def inbound():
    # Adding a delay so that all bots don't answer at once (could overload the API).
    # This will randomly choose a value between 0 and 10 using a uniform distribution.
    delay = random.uniform(0, 10)
    time.sleep(delay)
    req = request

    print '========POST REQUEST @ /slack========='
    response = {'username': my_bot_name, 'icon_emoji': ':robot_face:', 'text': ''}
    print 'FORM DATA RECEIVED IS:'
    print request.form

    channel = request.form.get('channel_name') #this is the channel name where the message was sent from
    username = request.form.get('user_name') #this is the username of the person who sent the message
    text = request.form.get('text') #this is the text of the message that was sent
    inbound_message = username + " in " + channel + " says: " + text
    print '\n\nMessage:\n' + inbound_message

    if username in [my_slack_username, 'zacharymulhaul'] or username == 'zac.wentzell':
            text == "<BOTS_RESPOND>"
            response['text'] = 'Hello, my name is zachgraysonbot. I belong to zacharymulhaul. I live at 41953/slack'
            print response['text']

    if username in [my_slack_username, 'zacharymulhaul'] or username == 'zac.wentzell':
            text == "<I_NEED_HELP_WITH CODING>:"
            # need to configure integrating =title value in text


            #Slack API component for Questions, etc.
            base_url = 'https://api.stackexchange.com/2.2/search/advanced?order=desc&sort=activity&answers&'
            complete_url = 'base_url + title + =5&site=stackoverflow'

            get_url = requests.get(complete_url).json()
            print(get_url)

            for item in get_url['items']:
                print item['title'],['link'],['creation_date']






    if slack_inbound_url and response['text']:
            r = requests.post(slack_inbound_url, json=response)

    print '========REQUEST HANDLING COMPLETE========\n\n'

    return Response(), 200


# this handles GET requests sent to your server at SERVERIP:41953/
@application.route('/', methods=['GET'])
def test():
    return Response('Your flask app is running!')


if __name__ == "__main__":
    application.run(host='0.0.0.0', port=41953)