# -*- coding: utf-8 -*-

import requests
import os

def send_line_message(message: str = "finish🎉") -> None:
    line_notify_token = os.environ.get("LINE_TOKEN")

    line_notify_api = "https://notify-api.line.me/api/notify"

    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {message}'}
    requests.post(line_notify_api, headers=headers, data=data)

if __name__ == "__main__":
    send_line_message("test")
